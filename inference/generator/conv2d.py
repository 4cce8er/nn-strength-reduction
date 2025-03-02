import numpy as np
from enum import Enum
from .layer import Layer
from .utils import to_np_dtype, multiply_by_quantize_mul
from generator.logger import logger

class Conv2D(Layer):
    n_layers = 0

    class Padding(Enum):
        SAME = 0
        VALID = 1

    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.id = Conv2D.n_layers
        Conv2D.n_layers += 1
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
            self.padding = Conv2D.Padding.SAME
        elif padding == "VALID":
            self.padding = Conv2D.Padding.VALID

    def compute_output_shape(self, input_size, filter_size, stride):
        out_size = 0
        match self.padding:
            case Conv2D.Padding.SAME:
                out_size = (input_size + stride - 1) // stride
            case Conv2D.Padding.VALID:
                out_size = (input_size - filter_size + stride) // stride
            case _:
                print("Invalid padding")
        return out_size

    def compute_padding(self, output_size, input_size, filter_size, stride):
        pad = ((output_size - 1) * stride + filter_size - input_size)
        offset = pad % 2
        if pad < 0:
            return 0
        return (pad // 2, offset)

    def pad_input(self, input: np.ndarray, pad_h: int, pad_w: int, pad_h_off: int, pad_w_off: int) -> np.ndarray:
        in_h = input.shape[1]
        in_w = input.shape[2]

        padded_h = in_h + 2*pad_h + pad_h_off
        padded_w = in_w + 2*pad_w + pad_w_off
        padded_input = np.zeros((1, padded_h, padded_w, input.shape[3]), dtype=np.int8)
        # padded_input[0, pad_h:pad_h+in_h, pad_w:pad_w+in_w, :] = input
        for i in range(in_h):
            for j in range(in_w):
                padded_input[0, pad_h+i, pad_w+j, :] = input[0, i, j, :]
        return padded_input

    def conv_by_filter(self, input: np.ndarray, filter: np.ndarray, ch_idx: int, fixed_point: dict, output_zero_point: np.int8, feature_map: np.ndarray) -> np.ndarray:
        logger.debug(f"input shape: {input.shape} - filter shape: {filter.shape} - stride: {self.stride_h} - {self.stride_w}")
        # out_height = input.shape[1] - filter.shape[1] + 1
        # out_height = np.floor((input.shape[1] + 2*0 - filter.shape[1]) / self.stride_h).astype(np.int32)
        if self.padding == Conv2D.Padding.VALID:
            pad_h = 0
            pad_w = 0
        else:
            pad_h, pad_h_off = self.compute_padding(self.output_shape[1], input.shape[1], filter.shape[1], self.stride_h)
            pad_w, pad_w_off = self.compute_padding(self.output_shape[2], input.shape[2], filter.shape[2], self.stride_w)
        # out_height = self.compute_output_shape(input.shape[1], filter.shape[1], self.stride_h)
        # pad_h, pad_h_off = self.compute_padding(out_height, input.shape[1], filter.shape[1], self.stride_h)
        # # out_width = input.shape[2] - filter.shape[2] + 1
        # # out_width = np.floor((input.shape[2] + 2*0 - filter.shape[2]) / self.stride_w).astype(np.int32)
        # out_width = self.compute_output_shape(input.shape[2], filter.shape[2], self.stride_w)
        # pad_w, pad_w_off = self.compute_padding(out_width, input.shape[2], filter.shape[2], self.stride_w)
        # logger.info(f"output shape: {out_height} - {out_width}")
        # logger.info(f"padding (H,W): {(pad_h, pad_w)} - offset: {(pad_h_off, pad_w_off)}")
        # input = self.pad_input(input, pad_h, pad_w, pad_h_off, pad_w_off)
        # print("input shape after padding:", input.shape)
        # print("input after padding:\n", input[0, :, :, 0])
        filter_height = filter.shape[1]
        filter_width = filter.shape[2]
        filter_depth = filter.shape[3]
        # Code generation
        self.opt_conv_code = ""
        self.opt_conv_code += f"static int32_t apply_filter_{ch_idx}_convlayer{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int out_h, int out_w, int ch_idx) " + "{\n"
        self.opt_conv_code += f"    int32_t acc = 0;\n"
        # self.opt_conv_code = f"int32_t apply_filter_{ch_idx}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int ch_idx, int8_t feature_map[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}]) " + "{\n"
        # self.opt_conv_code += f"    for(int out_h = 0; out_h < {out_height}; out_h++)" + "{\n"
        # self.opt_conv_code += f"        for(int out_w = 0; out_w < {out_width}; out_w++)" + "{\n"
        # self.opt_conv_code += f"            int32_t acc = 0;\n"
        for out_h in range(self.output_shape[1]):
            for out_w in range(self.output_shape[2]):
                in_y_origin = (out_h * self.stride_h) - pad_h
                in_x_origin = (out_w * self.stride_w) - pad_w
                acc = np.int32(0)
                for in_chan in range(filter_depth):
                    for filter_h in range(filter_height):
                        in_y = in_y_origin + filter_h
                        if 0 <= in_y < input.shape[1]:
                            for filter_w in range(filter_width):
                                in_x = in_x_origin + filter_w
                                if 0 <= in_x < input.shape[2]:
                                    input_val = np.int32(input[0][in_y][in_x][in_chan])
                                    filter_val = np.int32(filter[ch_idx][filter_h][filter_w][in_chan])
                                    acc += input_val * filter_val
                                    # print(f"input[0][{out_h * self.stride_h + f_h}][{out_w * self.stride_w + f_w}][{cin}] * filter[{ch_idx}][{f_h}][{f_w}][{cin}]")
                                    # acc += np.int32(input[0][out_h * self.stride_h + filter_h][out_w * self.stride_w + filter_w][in_chan]) * np.int32(filter[ch_idx][filter_h][filter_w][in_chan])
                acc += self.bias[ch_idx]
                acc = multiply_by_quantize_mul(np.int64(acc), fixed_point["mantissa"], fixed_point["exponent"])
                acc += output_zero_point
                acc = np.clip(acc, -128, 127)
                feature_map[0][out_h][out_w][ch_idx] = acc

        for in_chan in range(filter_depth):
            for filter_h in range(filter_height):
                for filter_w in range(filter_width):
                    # self.opt_conv_code += f"    acc += input[0][out_h + {f_h}][out_w + {f_w}][{cin}] * {filter[ch_idx][f_h][f_w][cin]}; // filter[{ch_idx}][{f_h}][{f_w}][{cin}];\n"
                    if(filter[ch_idx][filter_h][filter_w][in_chan] == 0):
                        continue
                    elif(filter[ch_idx][filter_h][filter_w][in_chan] == 1):
                        self.opt_conv_code += f"    acc += input[0][out_h + {filter_h}][out_w + {filter_w}][{in_chan}];\n"
                    elif(filter[ch_idx][filter_h][filter_w][in_chan] < 0):
                        self.opt_conv_code += f"    acc += multiply_n{abs(filter[ch_idx][filter_h][filter_w][in_chan])}(input[0][out_h + {filter_h}][out_w + {filter_w}][{in_chan}]);\n"
                    else:
                        self.opt_conv_code += f"    acc += multiply_{filter[ch_idx][filter_h][filter_w][in_chan]}(input[0][out_h + {filter_h}][out_w + {filter_w}][{in_chan}]);\n"

        self.opt_conv_code += f"    return acc;\n"
        self.opt_conv_code += "}\n"
        self.code += self.opt_conv_code
        # self.opt_conv_code += f"            acc += bias[{ch_idx}];\n"
        # self.opt_conv_code += f"            acc = multiply_by_quantize_mul(acc, {fixed_point['mantissa']}, {fixed_point['exponent']});\n"
        # self.opt_conv_code += f"            acc += output_zero_point;\n"
        # self.opt_conv_code += f"            acc = acc > 127 ? 127 : acc;\n"
        # self.opt_conv_code += f"            acc = acc < -128 ? -128 : acc;\n"
        # self.opt_conv_code += f"            feature_map[0][out_h][out_w][{ch_idx}] = acc;\n"
        # self.opt_conv_code += "        }\n"
        # self.opt_conv_code += "    }\n"
        # self.opt_conv_code += "}\n"
        return feature_map

    def conv2d(self, input: np.ndarray) -> np.ndarray:
        input.flags.writeable = False # make input read-only
        # logger.info(f"input shape: {input.shape}, {len(input.shape)}, {self.input_shape}")
        # logger.info(f"filter shape: {self.filter.shape}")
        # logger.info(f"bias shape: {self.bias.shape}")
        # logger.info(f"output shape: {self.output_shape}")
        # logger.info(f"bias {self.bias}")
        gen_code = True
        feature_map = np.zeros(self.output_shape, dtype=np.int8)
        # assert dimensions
        assert(len(input.shape) == 4)
        assert(len(self.filter.shape) == 4)
        assert(len(self.bias.shape) == 1)
        assert(len(self.output_shape) == 4)
        # Extract input dimensions
        input_batch_size = input.shape[0]
        input_height = input.shape[1]
        input_width = input.shape[2]
        input_depth = input.shape[3]
        # Extract filter dimensions
        filter_number = self.filter.shape[0]
        filter_height = self.filter.shape[1]
        filter_width = self.filter.shape[2]
        filter_depth = self.filter.shape[3]
        # Extract output dimensions
        output_batch_size = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]
        output_channels = self.output_shape[3]
        # assert dimensions
        assert(output_channels == filter_number)
        assert(filter_depth == input_depth)
        assert(input_batch_size == output_batch_size and input_batch_size == 1) # only single core batch execution
        # print(input.shape, input)
        logger.debug(self.filter)
        # Since we asserted that we have a single batch size, we can iterate over the output channels
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
        # print(self.code)
        return feature_map
    
    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.conv2d(input)

    def generate_code(self):
        '''Generate C code for the convolution layer'''
        code = ""
        code += f"void conv2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
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
        code += f"    const int32_t fixed_points[{self.filter.shape[0]}][2] = " + "{\n"
        for i in range(self.filter.shape[0]):
            code += f"        {{ {self.fixed_points[i]['mantissa']}, {self.fixed_points[i]['exponent']} }},\n"
        code += "    };\n"
        code += f"    for(int cout = 0; cout < {self.output_shape[3]}; cout++)" + "{\n"
        code += f"        for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
        code += f"            for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
        code += f"                int32_t acc = 0;\n"
        code += f"                for(int f_h = 0; f_h < {self.filter.shape[1]}; f_h++)" + "{\n"
        code += f"                    for(int f_w = 0; f_w < {self.filter.shape[2]}; f_w++)" + "{\n"
        code += f"                        for(int f_d = 0; f_d < {self.filter.shape[3]}; f_d++)" + "{\n"
        code += f"                            acc += (int32_t)input[0][out_h + f_h][out_w + f_w][f_d] * (int32_t)filter[cout][f_h][f_w][f_d];\n"
        code += "                        }\n"
        code += "                    }\n"
        code += "                 }\n"
        code += f"                acc += bias[cout];\n"
        code += f"                acc = multiply_by_quantize_mul(acc, fixed_points[cout][0], fixed_points[cout][1]);\n"
        code += f"                acc += output_zero_point;\n"
        code += f"                acc = acc > 127 ? 127 : acc;\n"
        code += f"                acc = acc < -128 ? -128 : acc;\n"
        code += f"                output[0][out_h][out_w][cout] = acc;\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n"
        return code

    def generate_opt_code(self):
        code = self.code
        code += f"void conv2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    int8_t filter[{self.filter.shape[0]}][{self.filter.shape[1]}][{self.filter.shape[2]}][{self.filter.shape[3]}] = " + "{\n"
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
        code += f"    const int32_t fixed_points[{self.filter.shape[0]}][2] = " + "{\n"
        for i in range(self.filter.shape[0]):
            code += f"        {{ {self.fixed_points[i]['mantissa']}, {self.fixed_points[i]['exponent']} }},\n"
        code += "    };\n"
        # code += f"    const int32_t q_mantissa = {self.q_mantissa};\n"
        # code += f"    const int32_t exponent = {self.exponent};\n"
        for i in range(self.output_shape[3]):
            code += f"    for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
            code += f"        for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
            code += f"            int32_t acc = apply_filter_{i}_convlayer{self.id}(input, out_h, out_w, {i});\n"
            code += f"            acc += bias[{i}];\n"
            code += f"            acc = multiply_by_quantize_mul(acc, fixed_points[{i}][0], fixed_points[{i}][1]);\n"
            code += f"            acc += output_zero_point;\n"
            code += f"            acc = acc > 127 ? 127 : acc;\n"
            code += f"            acc = acc < -128 ? -128 : acc;\n"
            code += f"            output[0][out_h][out_w][{i}] = acc;\n"
            code += "        }\n"
            code += "    }\n"
        code += "}\n"
        return code

    def print_filters(self):
        n_filters = self.filter.shape[0]
        for filter_idx in range(n_filters):
            filter = self.filter[filter_idx]
            print(f"Filter {filter_idx}:")
            for i in range(filter.shape[0]):
                for j in range(filter.shape[1]):
                    print(filter[i][j])
                print()