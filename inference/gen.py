#! /usr/bin/env python3
from generator.conv2d import Conv2D
from generator.depth_conv2d import DepthwiseConv2D
from generator.model import Model
from generator.layer import Layer
from generator.reshape import Reshape
from generator.avgpool import AvgPool2D
from generator.fully_connected import FullyConnected
from generator.maxpool import MaxPool2D
from generator.quantize import Quantize
from generator.logger import logger

import json
import numpy as np
import argparse

JSON_FILE = "mlp.json"
OUTPUT_PATH="mlp_generated"

def load_json(input_file: str):
    with open(input_file, "r") as f:
        return json.load(f)

def create_model(model: Model, extracted: dict):
    for layer in extracted["layers"]:
        if layer["type"] == "CONV_2D":
            conv2d = Conv2D(layer["input_shape"], layer["output_shape"])
            fixed_point = layer["fixed_point"]
            # conv2d.set_fixed_point(fixed_point["mantissa"], fixed_point["exponent"])
            conv2d.set_fixed_points(fixed_point)
            conv2d.set_zero_points(layer["z_input"], layer["z_weight"], layer["z_bias"], layer["z_output"])
            conv2d.set_filter(layer["weights"]["data"], layer["weights"]["dtype"])
            conv2d.set_bias(layer["bias"]["data"], layer["bias"]["dtype"])
            conv2d.set_stride(layer["stride"][0], layer["stride"][1])
            conv2d.set_padding(layer["padding"])
            # layers.append(conv2d)
            model.add_layer(conv2d)
        elif layer["type"] == "DEPTHWISE_CONV_2D":
            depthwise_conv2d = DepthwiseConv2D(layer["input_shape"], layer["output_shape"])
            fixed_point = layer["fixed_point"]
            depthwise_conv2d.set_fixed_points(fixed_point)
            depthwise_conv2d.set_zero_points(layer["z_input"], layer["z_weight"], layer["z_bias"], layer["z_output"])
            depthwise_conv2d.set_filter(layer["weights"]["data"], layer["weights"]["dtype"])
            depthwise_conv2d.set_bias(layer["bias"]["data"], layer["bias"]["dtype"])
            depthwise_conv2d.set_stride(layer["stride"][0], layer["stride"][1])
            depthwise_conv2d.set_padding(layer["padding"])
            # layers.append(depthwise_conv2d)
            model.add_layer(depthwise_conv2d)
        elif layer["type"] == "MAX_POOL_2D":
            maxpool2d = MaxPool2D(layer["input_shape"], layer["output_shape"])
            # layers.append(maxpool2d)
            model.add_layer(maxpool2d)
        elif layer["type"] == "AVERAGE_POOL_2D":
            avgpool2d = AvgPool2D(layer["input_shape"], layer["output_shape"])
            avgpool2d.set_filter_size(layer["filter_size"][0], layer["filter_size"][1])
            avgpool2d.set_stride(layer["stride"][0], layer["stride"][1])
            # layers.append(avgpool2d)
            model.add_layer(avgpool2d)
        elif layer["type"] == "QUANTIZE":
            quantize = Quantize(layer["input_shape"], Z_input=np.int8(-128), S_input=layer["s_input"])
            # layers.append(quantize)
            model.add_layer(quantize)
        elif layer["type"] == "RESHAPE":
            reshape = Reshape(layer["input_shape"], layer["output_shape"])
            # layers.append(reshape)
            model.add_layer(reshape)
        elif layer["type"] == "FULLY_CONNECTED":
            fc = FullyConnected(layer["input_shape"], layer["output_shape"])
            fixed_point = layer["fixed_point"]
            fc.set_fixed_point(fixed_point["mantissa"], fixed_point["exponent"])
            fc.set_zero_points(layer["z_input"], layer["z_weight"], layer["z_bias"], layer["z_output"])
            fc.set_weights(layer["weights"]["data"], layer["weights"]["dtype"])
            fc.set_bias(layer["bias"]["data"], layer["bias"]["dtype"])
            # layers.append(fc)
            model.add_layer(fc)
        else:
            print("Unknown layer type:", layer["type"])


def mnist_inference(model: Model):
    import keras.datasets.mnist as mnist
    mnist_train, mnist_test = mnist.load_data()
    (x_train, y_train) = mnist_train
    (x_test, y_test) = mnist_test

    model.check_image(x_test, y_test, 8)

    model.generate_code()
    model.generate_opt_code()

def speech_inference(model: Model):
    '''
    Audio file
    '''
    values = [
        -25, 70, -23, -13, -47, 127, -54, 108, -87, 77, 58, 43, 114, 123, 99, -58, -4, 66, -44, 120, -101, 104, 103, 13, -10, -38, -82, -29, -77, 31, 73, 26, -26, -78, -115, 55, -79, -40, 35, -38, -91, -35, -123, -105, -40, 105, -34, 84, 43, 50, 77, 70, 27, 52, -44, -111, -114, 2, -12, -63, -95, -67, 92, 7, -16, 105, -66, 33, -63, 97, 124, -25, -66, -127, -2, 23, 106, 92, -21, 22, 15, -72, -36, -86, 108, 48, -69, 123, -78, 47, -68, -44, 108, -104, 91, -36, -126, -102, 126, -61, 123, 122, 42, -70, 123, -87, 81, 102, -123, -68, -4, 20, -11, 88, 62, -31, 9, 121, -36, 59, 40, 25, -113, 21, 49, 107, 113, 51, -123, 111, 119, -128, 105, 33, -70, 101, 74, -117, 75, 80, -56, -57, -28, 61, -97, -93, -98, 40, -100, -5, -28, 69, -108, -13, -38, 69, -34, -53, -7, -29, -69, -16, -28, -92, -111, 30, -119, 92, 42, 84, 44, 114, -101, -112, 47, -69, -77, 77, 99, -48, -56, -57, -107, -36, 59, -17, -94, -103, 58, 27, -3, 117, -117, 97, -102, -100, -1, -93, 120, -87, 120, 36, -101, -109, 53, 74, -50, 104, 24, -78, -72, 96, -7, -51, -67, -76, 60, -33, -50, -9, 122, 75, -20, -123, 44, 6, -95, -85, 42, -102, -43, 34, 62, -16, 53, -13, -69, -124, -36, 83, -74, 20, 51, 47, 98, 112, 100, 30, -49, -78, -107, -55, 125, 2, -50, 41, -120, -16, 84, 50, 10, -87, -44, -56, 26, -118, 60, 85, -114, -104, 40, -60, 44, -37, 115, 14, -52, 87, -83, 27, -119, -62, 101, -122, 68, -77, 47, 77, 35, 4, -1, -83, 45, 84, -10, -57, 94, -78, -100, 108, -54, 68, -80, 118, -96, -93, 5, -20, 123, 50, -121, -124, 116, 108, -117, 57, -96, 58, 6, 67, -66, -123, 113, 108, 89, -25, -77, 55, 25, -48, 35, 99, -108, 83, 89, -76, 119, -34, 32, 114, -112, 40, 118, -123, 20, -127, 62, 52, 60, -60, -8, 122, -55, -23, 102, -93, 80, -102, 90, -23, -22, -2, -52, -2, -47, -91, 51, -56, 4, -45, -70, 20, 123, -79, 25, 16, -78, -41, -60, 110, 27, 60, 105, 101, -91, 79, -120, 117, 105, 98, -34, -45, -32, 42, 82, 50, 80, 5, 122, -44, 88, -75, 104, 84, -26, 2, -28, 24, 89, 40, 7, -11, -27, -16, -38, 10, -65, -30, 0, -87, -60, 94, -4, 37, 9, -50, -41, -39, 83, -47, 45, 44, 6, 21, 0, 108, -105, 100, 5, 113, 12, -116, -26, 113, -4, 64, -4, 59, -94, 124, 100, -26, 90, -31, -117, -29, 47, -30, 60, 3, 52, -23, -81, -70, 127, 47, -89, -106, 19, 44, -121, -97, 56, -19, -111, -76, -83, 13, 111, -49, 9, 84, 54, -29, -75, 65, 71, 100, -92, 3
    ]

    np_array = np.array(values).reshape(1, 490)
    # np_array = np_array.reshape(1, 49, 10, 1)
    # print(np_array)
    print(model.layers)
    # pause = input("Press enter to continue")
    for layer in model.layers:
        logger.info(f"Applying layer {layer.__class__.__name__}")
        np_array = layer.apply_layer(np_array)
    # print(np_array)

    for layer in model.layers:
        layer.apply_layer(np.zeros(layer.input_shape))

    model.generate_code()
    model.generate_opt_code()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="extracted.json",
        help="JSON extracted from TFLite"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="code",
        help="Generates the C code for the emulation"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["mnist", "speech"],
        default="mnist",
        help="Dataset for which the model was trained"
    )
    args = parser.parse_args()
    extracted = load_json(args.input)
    model = Model(output_path=args.output + "_generated")
    create_model(model, extracted)
    choice = args.dataset
    if choice == "mnist":
        mnist_inference(model)
    elif choice == "speech":
        speech_inference(model)
    else:
        print("Unknown dataset")
        return 1

if __name__ == "__main__":
    main()





# extracted = load_json()

# model = Model(output_path=OUTPUT_PATH)
# create_model(model)

# '''
# Audio file
# '''
# # values = [
# #     -25, 70, -23, -13, -47, 127, -54, 108, -87, 77, 58, 43, 114, 123, 99, -58, -4, 66, -44, 120, -101, 104, 103, 13, -10, -38, -82, -29, -77, 31, 73, 26, -26, -78, -115, 55, -79, -40, 35, -38, -91, -35, -123, -105, -40, 105, -34, 84, 43, 50, 77, 70, 27, 52, -44, -111, -114, 2, -12, -63, -95, -67, 92, 7, -16, 105, -66, 33, -63, 97, 124, -25, -66, -127, -2, 23, 106, 92, -21, 22, 15, -72, -36, -86, 108, 48, -69, 123, -78, 47, -68, -44, 108, -104, 91, -36, -126, -102, 126, -61, 123, 122, 42, -70, 123, -87, 81, 102, -123, -68, -4, 20, -11, 88, 62, -31, 9, 121, -36, 59, 40, 25, -113, 21, 49, 107, 113, 51, -123, 111, 119, -128, 105, 33, -70, 101, 74, -117, 75, 80, -56, -57, -28, 61, -97, -93, -98, 40, -100, -5, -28, 69, -108, -13, -38, 69, -34, -53, -7, -29, -69, -16, -28, -92, -111, 30, -119, 92, 42, 84, 44, 114, -101, -112, 47, -69, -77, 77, 99, -48, -56, -57, -107, -36, 59, -17, -94, -103, 58, 27, -3, 117, -117, 97, -102, -100, -1, -93, 120, -87, 120, 36, -101, -109, 53, 74, -50, 104, 24, -78, -72, 96, -7, -51, -67, -76, 60, -33, -50, -9, 122, 75, -20, -123, 44, 6, -95, -85, 42, -102, -43, 34, 62, -16, 53, -13, -69, -124, -36, 83, -74, 20, 51, 47, 98, 112, 100, 30, -49, -78, -107, -55, 125, 2, -50, 41, -120, -16, 84, 50, 10, -87, -44, -56, 26, -118, 60, 85, -114, -104, 40, -60, 44, -37, 115, 14, -52, 87, -83, 27, -119, -62, 101, -122, 68, -77, 47, 77, 35, 4, -1, -83, 45, 84, -10, -57, 94, -78, -100, 108, -54, 68, -80, 118, -96, -93, 5, -20, 123, 50, -121, -124, 116, 108, -117, 57, -96, 58, 6, 67, -66, -123, 113, 108, 89, -25, -77, 55, 25, -48, 35, 99, -108, 83, 89, -76, 119, -34, 32, 114, -112, 40, 118, -123, 20, -127, 62, 52, 60, -60, -8, 122, -55, -23, 102, -93, 80, -102, 90, -23, -22, -2, -52, -2, -47, -91, 51, -56, 4, -45, -70, 20, 123, -79, 25, 16, -78, -41, -60, 110, 27, 60, 105, 101, -91, 79, -120, 117, 105, 98, -34, -45, -32, 42, 82, 50, 80, 5, 122, -44, 88, -75, 104, 84, -26, 2, -28, 24, 89, 40, 7, -11, -27, -16, -38, 10, -65, -30, 0, -87, -60, 94, -4, 37, 9, -50, -41, -39, 83, -47, 45, 44, 6, 21, 0, 108, -105, 100, 5, 113, 12, -116, -26, 113, -4, 64, -4, 59, -94, 124, 100, -26, 90, -31, -117, -29, 47, -30, 60, 3, 52, -23, -81, -70, 127, 47, -89, -106, 19, 44, -121, -97, 56, -19, -111, -76, -83, 13, 111, -49, 9, 84, 54, -29, -75, 65, 71, 100, -92, 3
# # ]

# # np_array = np.array(values).reshape(1, 490)
# # # np_array = np_array.reshape(1, 49, 10, 1)
# # # print(np_array)
# # print(model.layers)
# # pause = input("Press enter to continue")
# # for layer in model.layers:
# #     logger.error(f"Applying layer {layer.__class__.__name__}")
# #     np_array = layer.apply_layer(np_array)
# # print(np_array)

# # for layer in model.layers:
# #     layer.apply_layer(np.zeros(layer.input_shape))

# '''
# MNIST
# '''
# import keras.datasets.mnist as mnist
# mnist_train, mnist_test = mnist.load_data()
# (x_train, y_train) = mnist_train
# (x_test, y_test) = mnist_test

# model.check_image(x_test, y_test, 8)



# model.generate_code()
# model.generate_opt_code()

