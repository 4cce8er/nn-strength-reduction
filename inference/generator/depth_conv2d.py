import numpy as np
from enum import Enum
from generator.utils import to_np_dtype, multiply_by_quantize_mul
from generator.layer import Layer
from generator.logger import logger

class DepthwiseConv2D(Layer):
    n_layer = 0

    class Padding(Enum):
        SAME = 0
        VALID = 1
    
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.id = DepthwiseConv2D.n_layer
        DepthwiseConv2D.n_layer += 1
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.code = ""
    
    def set_fixed_point(self, mantissa: np.int32, exponent: np.int32):
        self.q_mantissa = mantissa
        self.exponent = exponent
    
    def set_fixed_points(self, fixed_points: list):
        self.fixed_points = fixed_points

    def set_zero_points(self, input_zero_point: np.int8, filter_zero_point: np.int8, bias_zero_point: np.int8, output_zero_point: np.int8):
        self.input_zero_point = input_zero_point
        self.filter_zero_point = filter_zero_point
        self.bias_zero_point = bias_zero_point
        self.output_zero_point = output_zero_point

    def set_filter(self, filter: list, dtype: str):
        self.filter = np.asarray(filter, dtype=to_np_dtype(dtype))
    
    def set_bias(self, bias: list, dtype: str):
        self.bias = np.asarray(bias, dtype=to_np_dtype(dtype))
    
    def set_stride(self, stride_h, stride_w):
        self.stride_h = stride_h
        self.stride_w = stride_w
    
    def set_padding(self, padding):
        if padding == "SAME":
            self.padding = DepthwiseConv2D.Padding.SAME
        elif padding == "VALID":
            self.padding = DepthwiseConv2D.Padding.VALID
    
    def compute_padding(self, output_size, input_size, filter_size, stride):
        pad = ((output_size - 1) * stride + filter_size - input_size)
        offset = pad % 2
        if pad < 0:
            return 0
        return (pad // 2, offset)
    
    def conv_by_filter(self, input: np.ndarray, filter: np.ndarray, ch_idx: int, fixed_point: dict, output_zero_point: np.int8, feature_map: np.ndarray) -> np.ndarray:
        logger.debug(f"input shape: {input.shape} - filter shape: {filter.shape} - stride: {self.stride_h} - {self.stride_w}")
        if self.padding == DepthwiseConv2D.Padding.VALID:
            pad_h = 0
            pad_w = 0
        else:
            pad_h, pad_h_off = self.compute_padding(self.output_shape[1], input.shape[1], filter.shape[1], self.stride_h)
            pad_w, pad_w_off = self.compute_padding(self.output_shape[2], input.shape[2], filter.shape[2], self.stride_w)
        filter_height = filter.shape[1]
        filter_width = filter.shape[2]
        filter_depth = filter.shape[3]
        # Code generation
        self.opt_conv_code = ""
        self.opt_conv_code += f"static int32_t apply_filter_{ch_idx}_depthwise_conv2d_{self.id}(int8_t input[{input.shape[0]}][{input.shape[1]}][{input.shape[2]}][{input.shape[3]}], int out_h, int out_w, int ch_idx)" + "{\n"
        self.opt_conv_code += f"    int32_t acc = 0;\n"

        for out_h in range(self.output_shape[1]):
            for out_w in range(self.output_shape[2]):
                for in_chan in range(filter_depth):
                    out_chan = in_chan
                    in_y_origin = (out_h * self.stride_h) - pad_h
                    in_x_origin = (out_w * self.stride_w) - pad_w
                    acc = np.int32(0)
                    for filter_h in range(filter_height):
                        in_y = in_y_origin + filter_h
                        if 0 <= in_y < input.shape[1]:
                            for filter_w in range(filter_width):
                                in_x = in_x_origin + filter_w
                                if 0 <= in_x < input.shape[2]:
                                    input_val = np.int32(input[0][in_y][in_x][in_chan])
                                    filter_val = np.int32(filter[0][filter_h][filter_w][in_chan])
                                    acc += input_val * filter_val
                    acc += self.bias[out_chan]
                    acc = multiply_by_quantize_mul(np.int64(acc), fixed_point["mantissa"], fixed_point["exponent"])
                    acc += output_zero_point
                    acc = np.clip(acc, -128, 127)
                    feature_map[0][out_h][out_w][out_chan] = acc

        for filter_h in range(filter_height):
            for filter_w in range(filter_width):
                if(filter[0][filter_h][filter_w][ch_idx] == 0):
                    continue
                elif(filter[0][filter_h][filter_w][ch_idx] == 1):
                    self.opt_conv_code += f"    acc += (int32_t)input[0][out_h + {filter_h}][out_w + {filter_w}][ch_idx];\n"
                elif(filter[0][filter_h][filter_w][ch_idx] < 0):
                    self.opt_conv_code += f"    acc += multiply_n{abs(filter[0][filter_h][filter_w][ch_idx])}(input[0][out_h + {filter_h}][out_w + {filter_w}][ch_idx]);\n"
                else:
                    self.opt_conv_code += f"    acc += multiply_{filter[0][filter_h][filter_w][ch_idx]}(input[0][out_h + {filter_h}][out_w + {filter_w}][ch_idx]);\n"
        self.opt_conv_code += "    return acc;\n"
        self.opt_conv_code += "}\n\n"
        self.code += self.opt_conv_code
        
        return feature_map
        
    def depthwise_conv2d(self, input: np.ndarray):
        input.flags.writeable = False
        gen_code = True
        feature_map = np.zeros(self.output_shape, dtype=np.int8)
        assert(len(input.shape) == 4)
        assert(len(self.filter.shape) == 4)
        assert(len(self.bias.shape) == 1)
        assert(len(self.output_shape) == 4)
        input_batch_size = input.shape[0]
        input_height = input.shape[1]
        input_width = input.shape[2]
        input_depth = input.shape[3]
        filter_number = self.filter.shape[0]
        filter_height = self.filter.shape[1]
        filter_width = self.filter.shape[2]
        filter_depth = self.filter.shape[3]
        output_batch_size = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]
        output_channels = self.output_shape[3]
        logger.debug(f"input shape: {input.shape} - filter shape: {self.filter.shape} - bias shape: {self.bias.shape} - output shape: {self.output_shape}")
        # assert(output_channels == filter_number)
        assert(filter_depth == input_depth)
        assert(input_batch_size == output_batch_size and input_batch_size == 1)
        assert(filter_depth == self.output_shape[3])

        for cout in range(output_channels):
            feature_map = self.conv_by_filter(input, self.filter, cout, self.fixed_points[cout], self.output_zero_point, feature_map)
        
        logger.debug(f"feature map shape: {feature_map.shape}")
        line_width = np.get_printoptions()['linewidth']
        threshold = np.get_printoptions()['threshold']
        np.set_printoptions(linewidth=np.inf, threshold=np.inf)
        for i in range(output_channels):
            f_mapi = feature_map[0,:,:,i]
            logger.debug(f_mapi)
        np.set_printoptions(linewidth=line_width, threshold=threshold)
        return feature_map

    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.depthwise_conv2d(input)


    def generate_code(self):
        '''Generate C code for the depthwise convolution layer'''
        code = ""
        code += f"void depthwise_conv2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    const int8_t filter[{self.filter.shape[0]}][{self.filter.shape[1]}][{self.filter.shape[2]}][{self.filter.shape[3]}] = " + "{\n"
        for i in range(self.filter.shape[0]):
            code += "        {"
            for j in range(self.filter.shape[1]):
                code += " {"
                for k in range(self.filter.shape[2]):
                    code += " {"
                    for l in range(self.filter.shape[3]):
                        code += f" {self.filter[i][j][k][l]},"
                    code += " },"
                code += " },"
            code += " },\n"
        code += "    };\n"
        code += f"    const int32_t bias[{self.bias.shape[0]}] = " + "{"
        for i in range(self.bias.shape[0]):
            code += f" {self.bias[i]},"
        code += " };\n"
        code += f"    const int8_t output_zero_point = {self.output_zero_point};\n"
        code += f"    const int8_t input_zero_point = {self.input_zero_point};\n"
        code += f"    const int8_t filter_zero_point = {self.filter_zero_point};\n"
        code += f"    const int8_t bias_zero_point = {self.bias_zero_point};\n"
        code += f"    const int32_t fixed_points[{self.filter.shape[3]}][2] = " + "{\n"
        for i in range(self.filter.shape[3]):
            code += f"        {{ {self.fixed_points[i]['mantissa']}, {self.fixed_points[i]['exponent']} }},\n"
        code += "    };\n"
        code += f"    for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
        code += f"        for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
        code += f"            for(int in_chan = 0; in_chan < {self.filter.shape[3]}; in_chan++)" + "{\n"
        code += f"                int32_t acc = 0;\n"
        code += f"                for(int filter_h = 0; filter_h < {self.filter.shape[1]}; filter_h++)" + "{\n"
        code += f"                    for(int filter_w = 0; filter_w < {self.filter.shape[2]}; filter_w++)" + "{\n"
        code += f"                        acc += (int32_t)input[0][out_h + filter_h][out_w + filter_w][in_chan] * (int32_t)filter[0][filter_h][filter_w][in_chan];\n"
        code += "                    }\n"
        code += "                }\n"
        code += f"                acc += bias[in_chan];\n"
        code += f"                acc = multiply_by_quantize_mul(acc, fixed_points[in_chan][0], fixed_points[in_chan][1]);\n"
        code += f"                acc += output_zero_point;\n"
        code += f"                acc = (int8_t)acc;\n"
        code += f"                output[0][out_h][out_w][in_chan] = acc;\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n\n"
        return code

    def generate_opt_code(self):
        code = self.code
        code += f"void depthwise_conv2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    const int8_t filter[{self.filter.shape[0]}][{self.filter.shape[1]}][{self.filter.shape[2]}][{self.filter.shape[3]}] = " + "{\n"
        for i in range(self.filter.shape[0]):
            code += "        {"
            for j in range(self.filter.shape[1]):
                code += " {"
                for k in range(self.filter.shape[2]):
                    code += " {"
                    for l in range(self.filter.shape[3]):
                        code += f" {self.filter[i][j][k][l]},"
                    code += " },"
                code += " },"
            code += " },\n"
        code += "    };\n"
        code += f"    const int32_t bias[{self.bias.shape[0]}] = " + "{"
        for i in range(self.bias.shape[0]):
            code += f" {self.bias[i]},"
        code += " };\n"
        code += f"    const int8_t output_zero_point = {self.output_zero_point};\n"
        code += f"    const int8_t input_zero_point = {self.input_zero_point};\n"
        code += f"    const int8_t filter_zero_point = {self.filter_zero_point};\n"
        code += f"    const int8_t bias_zero_point = {self.bias_zero_point};\n"
        code += f"    const int32_t fixed_points[{self.filter.shape[3]}][2] = " + "{\n"
        for i in range(self.filter.shape[3]):
            code += f"        {{ {self.fixed_points[i]['mantissa']}, {self.fixed_points[i]['exponent']} }},\n"
        code += "    };\n"
        for i in range(self.output_shape[3]): # output_channels = filter_depth
            code += f"    for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
            code += f"        for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
            code += f"            int32_t acc = apply_filter_{i}_depthwise_conv2d_{self.id}(input, out_h, out_w, {i});\n"
            code += f"            acc += bias[{i}];\n"
            code += f"            acc = multiply_by_quantize_mul(acc, fixed_points[{i}][0], fixed_points[{i}][1]);\n"
            code += f"            acc += output_zero_point;\n"
            code += f"            acc = (int8_t)acc;\n"
            code += f"            output[0][out_h][out_w][{i}] = acc;\n"
            code += "        }\n"
            code += "    }\n"
        code += "}\n\n"
        return code