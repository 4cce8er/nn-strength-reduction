import tensorflow as tf

import keras
import numpy as np
import pathlib
import argparse

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.optimizers import SGD
from keras.losses import MeanSquaredError, CategoricalCrossentropy

# My modules
#from dataset import *
from datasets.mnist import Mnist
from tflite_utils import TFLiteUtils
from logger.logger import logging
print(tf.__version__)

NET_TYPE="cnn"
MODEL_DIR = "./models"

keras.utils.set_random_seed(42)

x_train, y_train, x_test, y_test = Mnist.expand_dims().get_scaled_data()

#x_train = tf.expand_dims(x_train, axis=-1)  # -1 adds a dimension in the last position
#x_test = tf.expand_dims(x_test, axis=-1)

logging.info(x_train.shape)
logging.info(x_test.shape)
assert x_train[0].shape == (28, 28, 1)

def create_model() -> Sequential:
    model = Sequential(
        [
            Conv2D(filters=4, kernel_size=3, input_shape=(28, 28, 1), use_bias=False, activation="relu"),
            MaxPool2D(pool_size=2),
            Conv2D(filters=2, kernel_size=2, use_bias=False, activation="relu"),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    print(model.summary())
    return model


def train(model: Sequential) -> Sequential:
    print("Training...")
    batch_size = 100
    epochs = 30
    model.compile(
        optimizer=SGD(learning_rate=0.05),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    logging.info(x_train.shape)
    logging.info(y_train.shape)


    model.fit(
        x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )

    return model


def evaluate(model: Sequential) -> Sequential:
    logging.info("Evaluating...")
    score = model.evaluate(x=x_test, y=y_test)
    logging.warn(score)


def predict(model: Sequential) -> Sequential:
    logging.info("Showing the first 10 predictions")
    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = ["Predicted", "Original", "Confidence", "Match"]
    predictions = model.predict(x=x_test)
    for i in range(10):
        table.add_row(
            [
                np.argmax(predictions[i]),
                np.argmax(y_test[i]),
                np.max(predictions[i]),
                "✅" if np.argmax(predictions[i]) == np.argmax(y_test[i]) else "❌",
            ]
        )
    print(table)

def save_model(model: Sequential, format: str = "keras", model_name: str = ""):
    models_dir = pathlib.Path(MODEL_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    # Save in all formats
    match format:
        case "h5":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.h5", save_format="h5")
        case "tf":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.tf", save_format="tf")
        case "keras":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.keras", save_format="keras")
        case "all":  # save all
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.h5", save_format="h5")
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.keras", save_format="keras")
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.tf", save_format="tf")


def load_model(path: str = f"{MODEL_DIR}/{NET_TYPE}_model.keras") -> Sequential:
    model = keras.models.load_model(path)
    return model

def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI utility that enables training and evaluation. The model is always saved.")
    parser.add_argument("-l", "--load", help="Load a saved model")
    parser.add_argument("-t", "--train", action="store_true", help="Train the model (creates one or use the loaded one)")
    parser.add_argument("-e", "--eval", action="store_true", help="Evaluate the model and show the first 10 predictions.")
    parser.add_argument("--lite", help="Convert to TFLite and evaluate it. Available modes:[dyn,float16,int]")
    parser.add_argument("--summary", action="store_true", help="Show the model summary")
    parser.add_argument("--name", help="Model name (default: "")", default="")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    if args.load:
        model = load_model(args.load)
    else:
        model = create_model()
    if args.train:
        model = train(model)
    if args.eval:
        evaluate(model)
        predict(model)
    if args.lite:
        tf_utils = TFLiteUtils(model, args.lite, model_name=NET_TYPE+args.name)
        # tf_utils.set_dataset(x_train, y_train, x_test, y_test)
        tf_utils.set_dataset_v2(dataset=Mnist)
        tf_utils.convert_to_tflite(repr_dataset=Mnist.representative_data_gen, input_type=tf.uint8)
    if args.summary:
        model.summary()
        # convert_to_tflite(model, args.lite, model_name="")
    save_model(model, model_name=NET_TYPE+args.name)



if __name__ == "__main__":
    main()
