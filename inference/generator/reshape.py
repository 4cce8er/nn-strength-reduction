import numpy as np
from generator.layer import Layer
from generator.logger import logger

class Reshape(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.input_shape = tuple(input_shape)
        self.output_shape = output_shape
    
    def reshape(self, input: np.ndarray) -> np.ndarray:
        return input.reshape(self.output_shape)
        # print(input.shape, self.input_shape, self.output_shape)
        if input.shape == self.input_shape:
            return input.reshape(self.output_shape)
        reshaped = np.zeros(self.output_shape, dtype=np.int8)
        logger.info(f"reshaped shape: {reshaped.shape}")
        logger.info(f"input shape: {input.shape}")
        logger.info(f"self.input shape: {self.input_shape}")
        logger.info(f"output shape: {self.output_shape}")
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                for k in range(self.input_shape[2]):
                    reshaped[0][i][j][k] = input[0][i][j][k]
        return reshaped
    
    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.reshape(input)
    
    def generate_code(self):
        return super().generate_code()
