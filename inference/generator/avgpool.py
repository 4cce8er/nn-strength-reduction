from generator.layer import Layer
from generator.logger import logger
import numpy as np

class AvgPool2D(Layer):
    n_layers = 0
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.id = AvgPool2D.n_layers
        AvgPool2D.n_layers += 1
        self.input_shape = input_shape
        self.output_shape = output_shape

    def set_filter_size(self, filter_h, filter_w):
        self.filter_h = filter_h
        self.filter_w = filter_w

    def set_stride(self, stride_h, stride_w):
        self.stride_h = stride_h
        self.stride_w = stride_w

    def avg_pool2d(self, input: np.ndarray) -> np.ndarray:
        output_batch = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]
        output_depth = self.output_shape[3]
        avgpooled = np.zeros(self.output_shape, dtype=np.int8)

        for batch in range(output_batch):
            for out_depth in range(output_depth):
                for out_h in range(output_height):
                    for out_w in range(output_width):
                        sum = 0
                        for i in range(self.filter_h):
                            for j in range(self.filter_w):
                                sum += input[batch][out_h*self.stride_h + i][out_w*self.stride_w + j][out_depth]
                        avgpooled[batch][out_h][out_w][out_depth] = sum // (self.filter_h * self.filter_w)
        logger.debug(avgpooled)
        return avgpooled

    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.avg_pool2d(input)

    ### Generate C code for the average pooling layer
    
    def generate_code(self):
        '''Generate C code for the average pooling layer'''
        code = f"void avg_pool2d_{self.id}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + " {\n"
        code += f"    int sum;\n"
        code += f"    for (int batch = 0; batch < {self.output_shape[0]}; batch++)" + " {\n"
        code += f"        for (int out_depth = 0; out_depth < {self.output_shape[3]}; out_depth++)" + " {\n"
        code += f"            for (int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + " {\n"
        code += f"                for (int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + " {\n"
        code += f"                    sum = 0;\n"
        code += f"                    for (int i = 0; i < {self.filter_h}; i++)" + " {\n"
        code += f"                        for (int j = 0; j < {self.filter_w}; j++)" + " {\n"
        code += f"                            sum += input[batch][out_h*{self.stride_h} + i][out_w*{self.stride_w} + j][out_depth];\n"
        code += "                        }\n"
        code += "                    }\n"
        code += f"                    output[batch][out_h][out_w][out_depth] = sum / ({self.filter_h} * {self.filter_w});\n"
        code += "                }\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n"
        return code
    
    def generate_opt_code(self):
        return self.generate_code()