import numpy as np
from prettytable import PrettyTable
from generator.layer import Layer
from generator.conv2d import Conv2D
from generator.maxpool import MaxPool2D
from generator.quantize import Quantize
from generator.fully_connected import FullyConnected
from generator.depth_conv2d import DepthwiseConv2D
from generator.logger import logger
import os


class Model:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        # Create output directory if it doesn't exist
        os.makedirs(output_path + "/include", exist_ok=True)
        os.makedirs(output_path + "/src", exist_ok=True)
        self.layers: list[Layer] = []
        self.source = ""
        self.header = ""
    
    def add_layer(self, layer: Layer):
        self.layers.append(layer)
    
    def predict(self, x_test: np.ndarray, y_test: np.ndarray):
        table = PrettyTable()
        table.field_names = ["Predicted", "Original", "Confidence", "Match"]
        for i in range(10):
            predictions = x_test[i]
            predictions = predictions.reshape(1, 28, 28, 1)
            for layer in self.layers:
                predictions = layer.apply_layer(predictions)
            table.add_row(
                [
                    np.argmax(predictions),
                    y_test[i],
                    np.max(predictions),
                    "✅" if np.argmax(predictions) == y_test[i] else "❌",
                ]
            )
        print(table)
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        correct = 0
        for i in range(len(x_test)):
            predictions = x_test[i]
            predictions = predictions.reshape(1, 28, 28, 1)
            for layer in self.layers:
                predictions = layer.apply_layer(predictions)
            if np.argmax(predictions) == y_test[i]:
                correct += 1
        print("Accuracy:", correct / len(x_test))

    def check_image(self, x_test: np.ndarray, y_test: np.ndarray, index: int):
        predictions = x_test[index]
        predictions = predictions.reshape(1, 28, 28, 1)
        for layer in self.layers:
            print(layer.__class__.__name__)
            predictions = layer.apply_layer(predictions)
        print(predictions)
        print("Predicted:", np.argmax(predictions))
        print("Original:", y_test[index])

    def generate_header(self, opt: bool = False):
        if opt == False:
            self.header = "#ifndef MODEL_H\n#define MODEL_H\n"
        else:
            self.header = "#ifndef MODEL_OPT_H\n#define MODEL_OPT_H\n"
        self.header += "#include <stdint.h>\n"
        self.header += "#include <stdlib.h>\n\n"
        self.header += "typedef union byte {\n"
        for layer in self.layers:
            if isinstance(layer, Quantize):
                tmp = 1
                for shape in layer.input_shape:
                    tmp *= shape
                self.header += "    uint8_t u8[{}];\n".format(tmp)
                self.header += "    int8_t i8[{}];\n".format(tmp)
                break
        self.header += "} byte_t;\n\n"
        for layer in self.layers:
                if isinstance(layer, Conv2D):
                    self.header += f"void conv2d_{layer.id}(int8_t input[{layer.input_shape[0]}][{layer.input_shape[1]}][{layer.input_shape[2]}][{layer.input_shape[3]}], int8_t output[{layer.output_shape[0]}][{layer.output_shape[1]}][{layer.output_shape[2]}][{layer.output_shape[3]}]);\n"
                elif isinstance(layer, DepthwiseConv2D):
                    self.header += f"void depthwise_conv2d_{layer.id}(int8_t input[{layer.input_shape[0]}][{layer.input_shape[1]}][{layer.input_shape[2]}][{layer.input_shape[3]}], int8_t output[{layer.output_shape[0]}][{layer.output_shape[1]}][{layer.output_shape[2]}][{layer.output_shape[3]}]);\n"
                elif isinstance(layer, MaxPool2D):
                    self.header += f"void max_pool2d_{layer.id}(int8_t input[{layer.input_shape[0]}][{layer.input_shape[1]}][{layer.input_shape[2]}][{layer.input_shape[3]}], int8_t output[{layer.output_shape[0]}][{layer.output_shape[1]}][{layer.output_shape[2]}][{layer.output_shape[3]}]);\n"
                elif isinstance(layer, Quantize):
                    self.header += "void quantize(byte_t* image);\n"
                elif isinstance(layer, FullyConnected):
                    self.header += f"void fully_connected(int8_t input[{layer.input_shape[0]}][{layer.input_shape[1]}], int8_t output[{layer.output_shape[1]}]);\n"
                else:
                    continue
        self.header += "\n#endif\n"


    def generate_code(self):
        self.generate_header()
        self.source = "#include \"model.h\"\n\n"
        for layer in self.layers:
            logger.info(f"Generating code for {layer.__class__.__name__}")
            self.source += layer.generate_code()
            self.source += "\n"
        # Save model.h
        with open(f"{self.output_path}/include/model.h", "w") as f:
            f.write(self.header)
        # Save model.c
        with open(f"{self.output_path}/src/model.c", "w") as f:
            f.write(self.source)
    
    def generate_opt_code(self):
        self.generate_header(opt=True)
        self.source = "#include \"model_opt.h\"\n"
        self.source += "#include \"multiply_inlined.h\"\n\n"
        for layer in self.layers:
            self.source += layer.generate_opt_code()
            self.source += "\n"
        # Save model_opt.h
        with open(f"{self.output_path}/include/model_opt.h", "w") as f:
            f.write(self.header)
        # Save model.c
        with open(f"{self.output_path}/src/model_opt.c", "w") as f:
            f.write(self.source)
