import numpy as np
from generator.layer import Layer
from enum import Enum

class MaxPool2D(Layer):
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
            pad_h = 0
            pad_w = 0
        else:
            pad_h, pad_h_off = self.compute_padding(self.output_shape[1], self.input_shape[1], self.filter_h, self.stride_h)
            pad_w, pad_w_off = self.compute_padding(self.output_shape[2], self.input_shape[2], self.filter_w, self.stride_w)
        output_batch = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]
        output_depth = self.output_shape[3]
        maxpooled = np.zeros(self.output_shape, dtype=np.int8)

        for batch in range(output_batch):
            for out_h in range(output_height):
                for out_w in range(output_width):
                    for depth in range(output_depth):
                        input_w_point = out_w * self.stride_w - pad_w
                        input_h_point = out_h * self.stride_h - pad_h
                        filter_start_w = max(0, -input_w_point)
                        filter_start_h = max(0, -input_h_point)
                        filter_end_w = min(self.filter_w, self.input_shape[2] - input_w_point)
                        filter_end_h = min(self.filter_h, self.input_shape[1] - input_h_point)
                        max_val = np.iinfo(np.int8).min
                        for filter_y in range(filter_start_h, filter_end_h):
                            for filter_x in range(filter_start_w, filter_end_w):
                                in_x = input_w_point + filter_x
                                in_y = input_h_point + filter_y
                                max_val = np.max(max_val, input[batch, in_y, in_x, depth])
                        
                        maxpooled[batch, out_h, out_w, depth] = max_val
                        # maxpooled[batch][out_h][out_w][depth] = np.max(input[batch][out_h*2:out_h*2+2, out_w*2:out_w*2+2, depth])

        # for i in range(self.output_shape[0]):
        #     for j in range(self.output_shape[1]):
        #         for k in range(self.output_shape[2]):
        #             for l in range(self.output_shape[3]):
        #                 maxpooled[i][j][k][l] = np.max(input[i][j*2:j*2+2, k*2:k*2+2, l])
        line_width = np.get_printoptions()['linewidth']
        threshold = np.get_printoptions()['threshold']
        np.set_printoptions(linewidth=np.inf, threshold=np.inf)
        m1 = maxpooled[0,:,:,0]
        m2 = maxpooled[0,:,:,1]
        # print(m1)
        # print(m2)
        np.set_printoptions(linewidth=line_width, threshold=threshold)
        return maxpooled
    
    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.max_pool2d(input)
    
    def generate_code(self):
        '''Generate C code for the max pooling layer'''
        code = ""
        code += f"void max_pool2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    for(int i = 0; i < {self.output_shape[0]}; i++)" + "{\n"
        code += f"        for(int j = 0; j < {self.output_shape[1]}; j++)" + "{\n"
        code += f"            for(int k = 0; k < {self.output_shape[2]}; k++)" + "{\n"
        code += f"                for(int l = 0; l < {self.output_shape[3]}; l++)" + "{\n"
        code += f"                    int8_t max = input[i][j*2][k*2][l];\n"
        code += f"                    max = input[i][j*2][k*2+1][l] > max ? input[i][j*2][k*2+1][l] : max;\n"
        code += f"                    max = input[i][j*2+1][k*2][l] > max ? input[i][j*2+1][k*2][l] : max;\n"
        code += f"                    max = input[i][j*2+1][k*2+1][l] > max ? input[i][j*2+1][k*2+1][l] : max;\n"
        code += f"                    output[i][j][k][l] = max;\n"
        code += "                }\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n"
        return code

    def generate_code_v2(self):
        '''Generate C code for the max pooling layer'''
        code = ""
        code += f"void max_pool2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    for(int batch = 0; batch < {self.output_shape[0]}; batch++)" + "{\n"
        code += f"        for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
        code += f"            for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
        code += f"                for(int depth = 0; depth < {self.output_shape[3]}; depth++)" + "{\n"
        code += f"                    const int input_w_point = out_w * {self.stride_w} - {self.padding[1]};\n"
        code += f"                    const int input_h_point = out_h * {self.stride_h} - {self.padding[0]};\n"
        code += f"                    const int filter_start_w = (-input_w_point > 0) ? -input_w_point : 0;\n"
        code += f"                    const int filter_start_h = (-input_h_point > 0) ? -input_h_point : 0;\n"
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

    def generate_opt_code(self):
        return self.generate_code()

