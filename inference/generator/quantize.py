from generator.layer import Layer
import numpy as np

class Quantize(Layer):
    def __init__(self, input_shape, Z_input: np.int8 = np.int8(0), S_input: np.float64 = np.float64(0)):
        self.array: np.ndarray = np.ndarray([])
        self.input_shape = input_shape
        self.Z_input = Z_input
        self.S_input = S_input

    def quantize(self, array: np.ndarray):
        """Quantization formulas
        r = S(q - Z)
        q = r/S + Z
        """
        self.array = array
        quantized = np.zeros(self.array.shape, dtype=np.int8)
        match self.array.dtype:
            case np.uint8:
                for i in range(self.array.shape[1]):
                    for j in range(self.array.shape[2]):
                        quantized[0][i][j] = self.array[0][i][j] + self.Z_input
            case np.float32:
                for i in range(self.array.shape[1]):
                    for j in range(self.array.shape[2]):
                        quantized[0][i][j] = (self.array[0][i][j] / self.S_input) + self.Z_input
            case np.float64:
                for i in range(self.array.shape[1]):
                    for j in range(self.array.shape[2]):
                        quantized[0][i][j] = (self.array[0][i][j] / self.S_input) + self.Z_input
            case _:
                print("another array dtype:", self.array.dtype)
        return quantized

    def apply_layer(self, array: np.ndarray):
        return self.quantize(array)
    
    def generate_code(self):
        input_dim = 1
        for shape in self.input_shape:
            input_dim *= shape
        # TODO Generate Header files
        # code = "typedef union byte {\n"
        # code += f"    uint8_t u8[{input_dim}];\n"
        # code += f"    int8_t i8[{input_dim}];\n"
        # code += "} byte_t;\n\n"
        code = "void quantize(byte_t* image)" + "{\n"
        code += "    for(size_t i = 0; i < 784; i++)" + "{\n"
        code += f"        image->i8[i] = image->u8[i] + ({self.Z_input});\n"
        code += "    }\n"
        code += "}\n\n"
        code += "static int32_t multiply_by_quantize_mul(int64_t acc, int32_t q_mantissa, int32_t exp)" + "{\n"
        code += "    const int32_t reduced_mantissa = q_mantissa < 0x7FFF0000 ? ((q_mantissa + (1 << 15)) >> 16) : 0x7FFF;\n"
        code += "    const int64_t total_shifts = 15 - exp;\n"
        code += "    const int64_t round = (int64_t)(1) << (total_shifts - 1);\n"
        code += "    acc = acc * (int64_t)reduced_mantissa;\n"
        code += "    acc = acc + round;\n"
        code += "    int32_t result = acc >> total_shifts;\n"
        code += "    return result;\n"
        code += "}\n"
        return code
    
    def generate_opt_code(self):
        return self.generate_code()