import numpy as np
import tensorflow as tf
from enum import Enum

class Conv2D():
    n_layers = 0
    class Padding(Enum):
        SAME = 0
        VALID = 1

    def __init__(self, input_shape, output_shape):
        self.id = Conv2D.n_layers
        Conv2D.n_layers += 1
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.code = ""

    def set_filters(self, filters: np.ndarray):
        self.filters = tf.transpose(filters, perm=[3, 0, 1, 2])
        print(self.filters)
    
    def set_padding(self, padding):
        if padding == "SAME":
            self.padding = Conv2D.Padding.SAME
        elif padding == "VALID":
            self.padding = Conv2D.Padding.VALID
    
    def compute_padding(self, output_size, input_size, filter_size, stride):
        pad = ((output_size - 1) * stride + filter_size - input_size)
        offset = pad % 2
        if pad < 0:
            return 0
        return (pad // 2, offset)

    def set_strides(self, stride_h, stride_w):
        self.stride_h = stride_h
        self.stride_w = stride_w

    def conv_by_filter(self, input: np.ndarray, filter: np.ndarray, chan_idx: int, feature_map: np.ndarray):
        if self.padding == Conv2D.Padding.VALID:
            self.pad_h = 0
            self.pad_w = 0
        else:
            self.pad_h, pad_h_off = self.compute_padding(self.output_shape[1], input.shape[1], filter.shape[1], self.stride_h)
            self.pad_w, pad_w_off = self.compute_padding(self.output_shape[2], input.shape[2], filter.shape[2], self.stride_w)

        filter_height = filter.shape[1]
        filter_width = filter.shape[2]
        filter_depth = filter.shape[3]

        # for out_h in range(self.output_shape[1]):
        #     in_y_origin = (out_h * self.stride_h) - pad_h
        #     for out_w in range(self.output_shape[2]):
        #         in_x_origin = (out_w * self.stride_w) - pad_w
        #         accumulator = 0
        #         for filter_h in range(filter_height):
        #             in_y = in_y_origin + filter_h
        #             for filter_w in range(filter_width):
        #                 in_x = in_x_origin + filter_w
        #                 is_point_valid = in_y >= 0 and in_y < input.shape[1] and in_x >= 0 and in_x < input.shape[2]
        #                 if is_point_valid:
        #                     for in_channel in range(filter_depth):
        #                         input_val = input[0, in_y, in_x, in_channel]
        #                         filter_val = filter[chan_idx, filter_h, filter_w, in_channel]
        #                         accumulator += input_val * filter_val
        #         feature_map[0, out_h, out_w, chan_idx] = accumulator
        for out_h in range(self.output_shape[1]):
            for out_w in range(self.output_shape[2]):
                in_y_origin = (out_h * self.stride_h) - self.pad_h
                in_x_origin = (out_w * self.stride_w) - self.pad_w
                accumulator = 0
                for in_channel in range(filter_depth):
                    for filter_h in range(filter_height):
                        in_y = in_y_origin + filter_h
                        if 0 <= in_y < input.shape[1]:
                            for filter_w in range(filter_width):
                                in_x = in_x_origin + filter_w
                                if 0 <= in_x < input.shape[2]:
                                    input_val = input[0, in_y, in_x, in_channel]
                                    filter_val = filter[chan_idx, filter_h, filter_w, in_channel]
                                    accumulator += input_val * filter_val
                feature_map[0, out_h, out_w, chan_idx] = accumulator

        # Generate filter code
        self.filter_code = f"static int32_t apply_filter_{chan_idx}_convlayer{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], const int in_y_origin, const int in_x_origin, int ch_idx)" + "{\n"
        self.filter_code += f"    int32_t accumulator = 0;\n"
        for out_h in range(self.output_shape[1]):
            for out_w in range(self.output_shape[2]):
                in_y_origin = (out_h * self.stride_h) - self.pad_h
                in_x_origin = (out_w * self.stride_w) - self.pad_w
                for in_channel in range(filter_depth):
                    for filter_h in range(filter_height):
                        in_y = in_y_origin + filter_h
                        if 0 <= in_y < input.shape[1]:
                            for filter_w in range(filter_width):
                                in_x = in_x_origin + filter_w
                                if 0 <= in_x < input.shape[2]:
                                    filter_val = filter[chan_idx, filter_h, filter_w, in_channel]
                                    if filter_val == 0:
                                        continue
                                    elif filter_val == 1:
                                        self.filter_code += f"    accumulator += input[0][{in_y}][{in_x}][{in_channel}];\n"
                                    elif filter_val < 0:
                                        self.filter_code += f"    accumulator += multiply_n{abs(filter_val)}(input[0][{in_y}][{in_x}][{in_channel}]);\n"
                                    else:
                                        self.filter_code += f"    accumulator += multiply_{filter_val}(input[0][{in_y}][{in_x}][{in_channel}]);\n"
        self.filter_code += "    return accumulator;\n"
        self.filter_code += "}\n"
        self.code += self.filter_code
        return feature_map

    def conv2d(self, input: np.ndarray):
        # input.flags.writeable = False
        # The output shape is (batch, height, width, channels)
        feature_map = np.zeros(self.output_shape, dtype=np.float32) # XXX: This is a hack, we should use np.int8 in generation code
        # assert dimensions
        assert(len(input.shape) == 4)
        assert(len(self.filters.shape) == 4)
        assert(len(self.output_shape) == 4)
        # Extract input dimensions
        input_batch_size = input.shape[0]
        input_height = input.shape[1]
        input_width = input.shape[2]
        input_depth = input.shape[3]
        # Extract filter dimensions
        filter_number = self.filters.shape[0]
        filter_height = self.filters.shape[1]
        filter_width = self.filters.shape[2]
        filter_depth = self.filters.shape[3]
        # Extract output dimensions
        output_batch_size = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]
        output_channels = self.output_shape[3]
        # assert dimensions
        assert(output_channels == filter_number)
        assert(filter_depth == input_depth)
        assert(input_batch_size == output_batch_size and input_batch_size == 1) # only single core batch execution
        # Since we asserted that we have a single batch size, we can iterate over the output channels
        for cout in range(output_channels):
            feature_map = self.conv_by_filter(input, self.filters, cout, feature_map)
        return tf.convert_to_tensor(feature_map, dtype=tf.float32)

    def generate_code(self):
        '''Generate C code for the convolutional layer'''
        code = ""
        code += f"void conv2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    for(int out_ch = 0; out_ch < {self.output_shape[3]}; out_ch++)" + "{\n"
        code += f"        for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
        code += f"            for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
        code += f"                const int in_y_origin = (out_h * {self.stride_w}) - {self.pad_h};\n"
        code += f"                const int in_x_origin = (out_w * {self.stride_w}) - {self.pad_w};\n"
        code += f"                int32_t accumulator = 0;\n"
        code += f"                for(int in_ch = 0; in_ch < {self.input_shape[3]}; in_ch++)" + "{\n"
        code += f"                    for(int filter_h = 0; filter_h < {self.filters.shape[1]}; filter_h++)" + "{\n"
        code += f"                        const int in_y = in_y_origin + filter_h;\n"
        code += f"                        if(in_y >= 0 && in_y < {self.input_shape[1]})" + "{\n"
        code += f"                            for(int filter_w = 0; filter_w < {self.filters.shape[2]}; filter_w++)" + "{\n"
        code += f"                                const int in_x = in_x_origin + filter_w;\n"
        code += f"                                if(in_x >= 0 && in_x < {self.input_shape[2]})" + "{\n"
        code += f"                                    accumulator += input[0][in_y][in_x][in_ch] * filters[out_ch][filter_h][filter_w][in_ch];\n"
        code += "                                }\n"
        code += "                            }\n"
        code += "                        }\n"
        code += "                    }\n"
        code += f"                output[0][out_h][out_w][out_ch] = accumulator;\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n"
        return code



    def generate_opt_code(self):
        '''Generate optimized C code for the convolutional layer'''
        code = self.code
        code += f"void conv2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        for out_ch in range(self.output_shape[3]):
            code += f"    for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
            code += f"        for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
            code += f"            const int in_y_origin = (out_h * {self.stride_w}) - {self.pad_h};\n"
            code += f"            const int in_x_origin = (out_w * {self.stride_w}) - {self.pad_w};\n"
            code += f"            int32_t accumulator = apply_filter_{out_ch}_convlayer{self.id}(input, out_h, out_w, {out_ch});\n"
            code += f"            output[0][out_h][out_w][{out_ch}] = accumulator;\n"
            code += "        }\n"
            code += "    }\n"
        code += "}\n"
        return code


class DepthwiseConv2D():

    class Padding(Enum):
        SAME = 0
        VALID = 1
    
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def set_filters(self, filters: np.ndarray):
        self.filters = tf.transpose(filters, perm=[3, 0, 1, 2])
    
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
    
    def set_strides(self, stride_h, stride_w):
        self.stride_h = stride_h
        self.stride_w = stride_w

    def conv_by_filter(self, input: np.ndarray, filter: np.ndarray, chan_idx: int, feature_map: np.ndarray):
        if self.padding == DepthwiseConv2D.Padding.VALID:
            pad_h = 0
            pad_w = 0
        else:
            pad_h, pad_h_off = self.compute_padding(self.output_shape[1], input.shape[1], filter.shape[1], self.stride_h)
            pad_w, pad_w_off = self.compute_padding(self.output_shape[2], input.shape[2], filter.shape[2], self.stride_w)

        filter_height = filter.shape[1]
        filter_width = filter.shape[2]
        filter_depth = filter.shape[3]

        for out_h in range(self.output_shape[1]):
            for out_w in range(self.output_shape[2]):
                for in_channel in range(filter_depth):
                    out_channel = in_channel
                    in_y_origin = (out_h * self.stride_h) - pad_h
                    in_x_origin = (out_w * self.stride_w) - pad_w
                    accumulator = 0
                    for filter_h in range(filter_height):
                        in_y = in_y_origin + filter_h
                        if 0 <= in_y < input.shape[1]:
                            for filter_w in range(filter_width):
                                in_x = in_x_origin + filter_w
                                if 0 <= in_x < input.shape[2]:
                                    input_val = input[0, in_y, in_x, in_channel]
                                    filter_val = filter[0, filter_h, filter_w, in_channel]
                                    accumulator += input_val * filter_val
                    feature_map[0, out_h, out_w, out_channel] = accumulator
        return feature_map
    
    def depthwise_conv2d(self, input: np.ndarray):
        feature_map = np.zeros(self.output_shape, dtype=np.float32)
        # assert dimensions
        assert(len(input.shape) == 4)
        assert(len(self.filters.shape) == 4)
        assert(len(self.output_shape) == 4)
        # Extract input dimensions
        input_batch_size = input.shape[0]
        input_height = input.shape[1]
        input_width = input.shape[2]
        input_depth = input.shape[3]
        # Extract filter dimensions
        filter_number = self.filters.shape[0]
        filter_height = self.filters.shape[1]
        filter_width = self.filters.shape[2]
        filter_depth = self.filters.shape[3]
        # Extract output dimensions
        output_batch_size = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]
        output_channels = self.output_shape[3]
        # assert dimensions
        # FIXME assert(output_channels == filter_number) find a new assert for the depthwise
        assert(output_channels == filter_depth)
        assert(filter_depth == input_depth)
        assert(input_batch_size == output_batch_size and input_batch_size == 1)
        assert(self.filters.shape[3] == self.output_shape[3])
        # Since we asserted that we have a single batch size, we can iterate over the output channels
        for cout in range(output_channels):
            feature_map = self.conv_by_filter(input, self.filters, cout, feature_map)
        return tf.convert_to_tensor(feature_map, dtype=tf.float32)

class MaxPool2D():
    n_layers = 0

    class Padding(Enum):
        VALID = 0
        SAME = 1

    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.id = MaxPool2D.n_layers
        MaxPool2D.n_layers += 1
        self.input_shape = input_shape
        self.output_shape = output_shape

    def set_filter_size(self, filter_h, filter_w):
        self.filter_h = filter_h
        self.filter_w = filter_w

    def set_stride(self, stride_h, stride_w):
        self.stride_h = stride_h
        self.stride_w = stride_w

    def set_padding(self, padding):
        if padding == "VALID":
            self.padding = MaxPool2D.Padding.VALID
        elif padding == "SAME":
            self.padding = MaxPool2D.Padding.SAME
    
    def compute_padding(self, output_size, input_size, filter_size, stride):
        pad = ((output_size - 1) * stride + filter_size - input_size)
        offset = pad % 2
        if pad < 0:
            return 0
        return (pad // 2, offset)

    def max_pool2d(self, input: np.ndarray) -> np.ndarray:
        if self.padding == MaxPool2D.Padding.VALID:
            self.pad_h = 0
            self.pad_w = 0
        else:
            self.pad_h, pad_h_off = self.compute_padding(self.output_shape[1], self.input_shape[1], self.filter_h, self.stride_h)
            self.pad_w, pad_w_off = self.compute_padding(self.output_shape[2], self.input_shape[2], self.filter_w, self.stride_w)
        output_batch = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]
        output_depth = self.output_shape[3]
        maxpooled = np.zeros(self.output_shape, dtype=np.int8)
        for batch in range(output_batch):
            for out_h in range(output_height):
                for out_w in range(output_width):
                    input_w_point = out_w * self.stride_w - self.pad_w
                    input_h_point = out_h * self.stride_h - self.pad_h
                    filter_start_w = max(0, -input_w_point)
                    filter_start_h = max(0, -input_h_point)
                    filter_end_w = min(self.filter_w, self.input_shape[2] - input_w_point)
                    filter_end_h = min(self.filter_h, self.input_shape[1] - input_h_point)
                    print("################")
                    print(f"input_w_point = {input_w_point}")
                    print(f"input_h_point = {input_h_point}")
                    print(f"filter_start_w = {filter_start_w}")
                    print(f"filter_start_h = {filter_start_h}")
                    print(f"filter_end_w = {filter_end_w}")
                    print(f"filter_end_h = {filter_end_h}")
                    for depth in range(output_depth):

                        max_val = np.iinfo(np.int8).min
                        for filter_y in range(filter_start_h, filter_end_h):
                            for filter_x in range(filter_start_w, filter_end_w):
                                in_x = input_w_point + filter_x
                                in_y = input_h_point + filter_y
                                max_val = max(max_val, input[batch, in_y, in_x, depth])
                                # print(f"input[{batch}][{in_y}][{in_x}][{depth}] = {input[batch, in_y, in_x, depth]}")
                        
                        maxpooled[batch, out_h, out_w, depth] = max_val
                        # maxpooled[batch][out_h][out_w][depth] = np.max(input[batch][out_h*2:out_h*2+2, out_w*2:out_w*2+2, depth])
        # for i in range(self.output_shape[0]):
        #     for j in range(self.output_shape[1]):
        #         for k in range(self.output_shape[2]):
        #             for l in range(self.output_shape[3]):
        #                 maxpooled[i][j][k][l] = np.max(input[i][j*2:j*2+2, k*2:k*2+2, l])
        return maxpooled

    def max_pool2d_legacy(self, input: np.ndarray) -> np.ndarray:
        if self.padding == MaxPool2D.Padding.VALID:
            self.pad_h = 0
            self.pad_w = 0
        else:
            self.pad_h, pad_h_off = self.compute_padding(self.output_shape[1], self.input_shape[1], self.filter_h, self.stride_h)
            self.pad_w, pad_w_off = self.compute_padding(self.output_shape[2], self.input_shape[2], self.filter_w, self.stride_w)
        output_batch = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]
        output_depth = self.output_shape[3]
        maxpooled = np.zeros(self.output_shape, dtype=np.int8)
        for batch in range(output_batch):
            for out_h in range(output_height):
                for out_w in range(output_width):
                    for depth in range(output_depth):
                        max_val = input[batch, (out_h * self.stride_h) - self.pad_h, (out_w * self.stride_w) - self.pad_w, depth]
                        max_val = max(max_val, input[batch, (out_h * self.stride_h) - self.pad_h, (out_w * self.stride_w) - self.pad_w + 1, depth])
                        max_val = max(max_val, input[batch, (out_h * self.stride_h) - self.pad_h + 1, (out_w * self.stride_w) - self.pad_w, depth])
                        max_val = max(max_val, input[batch, (out_h * self.stride_h) - self.pad_h + 1, (out_w * self.stride_w) - self.pad_w + 1, depth])
                        maxpooled[batch, out_h, out_w, depth] = max_val
        return maxpooled

    def generate_code_v2(self):
        '''Generate C code for the max pooling layer'''
        code = ""
        code += f"void max_pool2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    for(int batch = 0; batch < {self.output_shape[0]}; batch++)" + "{\n"
        code += f"        for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
        code += f"            for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
        code += f"                for(int depth = 0; depth < {self.output_shape[3]}; depth++)" + "{\n"
        code += f"                    const int input_w_point = out_w * {self.stride_w} - {self.pad_w};\n"
        code += f"                    const int input_h_point = out_h * {self.stride_h} - {self.pad_h};\n"
        code += f"                    const int filter_start_w = max(0, -input_w_point);\n"
        code += f"                    const int filter_start_h = max(0, -input_h_point);\n"
        code += f"                    const int filter_end_w = min({self.filter_w}, {self.input_shape[2]} - input_w_point);\n"
        code += f"                    const int filter_end_h = min({self.filter_h}, {self.input_shape[1]} - input_h_point);\n"
        code += f"                    int8_t max_val = INT8_MIN;\n"
        code += f"                    for(int filter_y = filter_start_h; filter_y < filter_end_h; filter_y++)" + "{\n"
        code += f"                        for(int filter_x = filter_start_w; filter_x < filter_end_w; filter_x++)" + "{\n"
        code += f"                            const int in_x = input_w_point + filter_x;\n"
        code += f"                            const int in_y = input_h_point + filter_y;\n"
        code += f"                            max_val = max(max_val, input[batch][in_y][in_x][depth]);\n"
        code += "                        }\n"
        code += "                    }\n"
        code += f"                    output[batch][out_h][out_w][depth] = max_val;\n"
        code += "                }\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n"
        return code

    def generate_code_v3(self):
        '''Generate C code for the max pooling layer'''
        code = ""
        code += f"void max_pool2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    for(int i = 0; i < {self.output_shape[0]}; i++)" + "{\n"
        code += f"        for(int j = 0; j < {self.output_shape[1]}; j++)" + "{\n"
        code += f"            for(int k = 0; k < {self.output_shape[2]}; k++)" + "{\n"
        code += f"                for(int l = 0; l < {self.output_shape[3]}; l++)" + "{\n"
        code += f"                    int8_t max_val = INT8_MIN;\n"
        for batch in range(self.output_shape[0]):
            for out_h in range(self.output_shape[1]):
                for out_w in range(self.output_shape[2]):
                    input_w_point = out_w * self.stride_w - self.pad_w
                    input_h_point = out_h * self.stride_h - self.pad_h
                    filter_start_w = max(0, -input_w_point)
                    filter_start_h = max(0, -input_h_point)
                    filter_end_w = min(self.filter_w, self.input_shape[2] - input_w_point)
                    filter_end_h = min(self.filter_h, self.input_shape[1] - input_h_point)
                    for depth in range(self.output_shape[3]):
                        for filter_y in range(filter_start_h, filter_end_h):
                            for filter_x in range(filter_start_w, filter_end_w):
                                in_x = input_w_point + filter_x
                                in_y = input_h_point + filter_y
                                code += f"                    max_val = max(max_val, input[{batch}][{in_y}][{in_x}][{depth}]);\n"
        code += f"                    output[i][j][k][l] = max_val;\n"
        code += "                }\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n"
        return code


# Check code generation for Conv2D with N filters
input_shape = (1, 4, 4, 1)
output_shape = (1, 2, 2, 3)
conv2d = Conv2D(input_shape, output_shape)
filters = np.random.randint(-5, 5, (3, 3, 3, 1))
filters = tf.random.uniform((3, 3, 1, 3), minval=-5, maxval=5, dtype=tf.int32).numpy()
filters = np.array(
    [
        [
            [[-1], [-2], [-2]],
            [[ 4], [ 3], [ 3]],
            [[-3], [ 1], [-4]]
        ],
        [
            [[ 3], [-3], [-2]],
            [[ 1], [-3], [-3]],
            [[ 0], [-1], [ 3]]
        ],
        [
            [[-1], [-4], [ 4]],
            [[-1], [-4], [ 3]],
            [[-4], [ 3], [-5]]
        ]
    ],
    dtype=np.int32
)
print(filters.shape)
filters = filters.transpose(1, 2, 3, 0)
conv2d.set_filters(filters)
conv2d.set_padding("VALID")
conv2d.set_strides(2, 2)
input = np.random.randint(-5, 5, input_shape)
input_array = np.array(
    [
        [
            [[-3], [-3], [ 1], [-3]],
            [[-2], [-1], [ 2], [-3]],
            [[-3], [ 0], [ 4], [-4]],
            [[ 1], [ 2], [-3], [-5]]
        ]
    ],
    dtype=np.int32
)
print(input)
feature_map = conv2d.conv2d(input_array)
print(feature_map)
print(conv2d.generate_opt_code())
print(conv2d.generate_code())
