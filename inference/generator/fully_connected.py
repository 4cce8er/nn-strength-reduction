import numpy as np
from generator.layer import Layer
from generator.utils import to_np_dtype, multiply_by_quantize_mul

class FullyConnected(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def set_fixed_point(self, mantissa: np.int32, exponent: np.int32):
        self.q_mantissa = mantissa
        self.exponent = exponent
    
    def set_zero_points(self, input_zero_point: np.int8, weight_zero_point: np.int8, bias_zero_point: np.int8, output_zero_point: np.int8):
        self.input_zero_point = input_zero_point
        self.weight_zero_point = weight_zero_point
        self.bias_zero_point = bias_zero_point
        self.output_zero_point = output_zero_point
    
    def set_weights(self, weights: list, dtype: str):
        self.weights = np.asarray(weights, dtype=to_np_dtype(dtype))

    def set_bias(self, bias: list, dtype: str):
        self.bias = np.asarray(bias, dtype=to_np_dtype(dtype))
    
    def __dot__(self, input: np.ndarray) -> np.ndarray:
        # print(input.shape, self.weights.shape)
        result = np.zeros(self.output_shape, dtype=np.int32)
        for i in range(self.output_shape[0]):
            for j in range(self.weights.shape[0]):
                for k in range(self.weights.shape[1]):
                    result[i][j] += np.int32(input[i][k]) * np.int32(self.weights[j][k])
        # print(result.shape)
        return result

    def fully_connected(self, input: np.ndarray) -> np.ndarray:
        # print(input)
        dot_product = self.__dot__(input)
        dot_product += self.bias.transpose() # FIXME
        dot_product = multiply_by_quantize_mul(np.int64(dot_product), self.q_mantissa, self.exponent)
        dot_product += self.output_zero_point
        dot_product = np.clip(dot_product, -128, 127)
        dot_product = dot_product.astype(np.int8)
        return dot_product

    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.fully_connected(input)
    
    def generate_code(self, opt: bool = False):
        code = self.generate_dot() if opt == False else self.generate_opt_dot()
        code += f"void fully_connected(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}], int8_t output[{self.output_shape[1]}])" + "{\n"
        code += f"    const int8_t weights[{self.weights.shape[0]}][{self.weights.shape[1]}] = " + "{\n"
        for i in range(self.weights.shape[0]):
            code += "        {"
            for j in range(self.weights.shape[1]):
                code += f" {self.weights[i][j]},"
            code += " },\n"
        code += "    };\n"
        code += f"    const int32_t bias[{self.bias.shape[0]}] = " + "{"
        for i in range(self.bias.shape[0]):
            code += f" {self.bias[i][0]},"
        code += " };\n"
        code += f"    const int8_t output_zero_point = {self.output_zero_point};\n"
        code += f"    const int8_t input_zero_point = {self.input_zero_point};\n"
        code += f"    const int8_t weight_zero_point = {self.weight_zero_point};\n"
        code += f"    const int8_t bias_zero_point = {self.bias_zero_point};\n"
        code += f"    const int32_t q_mantissa = {self.q_mantissa};\n"
        code += f"    const int32_t exponent = {self.exponent};\n"
        code += f"    int32_t dot_result[{self.output_shape[1]}] = {{0}};\n"
        if opt == False:
            code += f"    dot_product(input, weights, dot_result);\n"
        else:
            code += f"    dot_product(input, dot_result);\n"
        code += f"    for(int i = 0; i < {self.output_shape[1]}; i++)" + "{\n"
        code += f"        dot_result[i] = dot_result[i] + bias[i];\n"
        code += f"        dot_result[i] = multiply_by_quantize_mul(dot_result[i], q_mantissa, exponent);\n"
        code += f"        dot_result[i] += output_zero_point;\n"
        code += f"        dot_result[i] = dot_result[i] > 127 ? 127 : dot_result[i];\n"
        code += f"        dot_result[i] = dot_result[i] < -128 ? -128 : dot_result[i];\n"
        code += f"        output[i] = dot_result[i];\n"
        code += "    }\n"
        code += "}\n"
        return code
    
    def generate_dot(self):
        code = ""
        code += f"static void dot_product(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}], const int8_t weights[{self.weights.shape[0]}][{self.weights.shape[1]}], int32_t dot_result[{self.weights.shape[0]}])" + "{\n"
        code += f"    for(int i = 0; i < {self.output_shape[0]}; i++)" + "{\n"
        code += f"        for(int j = 0; j < {self.weights.shape[0]}; j++)" + "{\n"
        code += f"            for(int k = 0; k < {self.weights.shape[1]}; k++)" + "{\n"
        code += f"                dot_result[j] += (int32_t)input[i][k] * (int32_t)weights[j][k];\n"
        code += "            }\n"
        code += "       }\n"
        code += "    }\n"
        code += "}\n"
        return code
    
    def generate_opt_dot(self):
        code = ""
        code += f"static void dot_product(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}], int32_t out[{self.output_shape[1]}])" + "{\n"
        for i in range(self.weights.shape[0]):
            code += f"    out[{i}] = 0;\n"
            for j in range(self.weights.shape[1]):
                if(self.weights[i][j] == 0):
                    continue
                elif(self.weights[i][j] == 1):
                    code += f"    out[{i}] += input[0][{j}];\n"
                elif(self.weights[i][j] < 0):
                    code += f"    out[{i}] += multiply_n{abs(self.weights[i][j])}(input[0][{j}]);\n"
                else:
                    code += f"    out[{i}] += multiply_{self.weights[i][j]}(input[0][{j}]);\n"
        code += "}\n"
        return code
    
    def generate_opt_code(self):
        return self.generate_code(opt=True)
