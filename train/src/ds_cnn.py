# from data_preprocessing import data_preprocessing
from data_preprocessing import constants
import tensorflow as tf
import numpy as np
import keras
import argparse
import pathlib
from logger.logger import logging
from tflite_utils import TFLiteUtils
from datasets.speech import SpeechCommand, model_settings

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
keras.utils.set_random_seed(seed)

MODEL_DIR = "./models"
NET_TYPE = "ds_cnn"

train_data, val_data, test_data = SpeechCommand.get_data()
audio_processor = SpeechCommand.get_audio_processor()


def tiny_ml_model_r2(model_settings):
    """
    This model is a revision of the model available on TinyML benchmarks
    https://github.com/mlcommons/tiny
    """
    input_shape = [
        model_settings["spectrogram_length"],
        model_settings["dct_coefficient_count"],
        1,
    ]
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]
    filters = 16
    weight_decay = 1e-4
    # regularizer = l2(weight_decay)
    regularizer = keras.regularizers.l2(l2=weight_decay)
    final_pool_size = (int(input_shape[0] / 2), int(input_shape[1] / 2))

    # Model layers
    # Input pure conv2d
    inputs = keras.Input(shape=(model_settings["fingerprint_size"],), name="input")
    x = keras.layers.Reshape((input_time_size, input_frequency_size, 1))(inputs)
    x = keras.layers.Conv2D(
        filters, (10, 4), strides=(1, 1), padding="valid", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=(3, 3),
        padding="valid",
        kernel_regularizer=regularizer,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), padding="valid", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # Second layer of separable depthwise conv2d
    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=(3, 3),
        padding="valid",
        kernel_regularizer=regularizer,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), padding="valid", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # Third layer of separable depthwise conv2d
    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=(3, 3),
        padding="valid",
        kernel_regularizer=regularizer,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), padding="valid", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # Fourth layer of separable depthwise conv2d
    # x = keras.layers.DepthwiseConv2D(
    #     depth_multiplier=1,
    #     kernel_size=(3, 3),
    #     padding="valid",
    #     kernel_regularizer=regularizer,
    # )(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation("relu")(x)
    # x = keras.layers.Conv2D(
    #     filters, (1, 1), padding="valid", kernel_regularizer=regularizer
    # )(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation("relu")(x)

    # Reduce size and apply final softmax
    x = keras.layers.Dropout(rate=0.4)(x)
    final_pool_size = (int(x.shape[1]), int(x.shape[2]))
    x = keras.layers.AveragePooling2D(pool_size=final_pool_size)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(model_settings["label_count"], activation="softmax")(x)

    # Instantiate model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def tiny_ml_model_r1(model_settings):
    """
    This model is a revision of the model available on TinyML benchmarks
    """
    input_shape = [
        model_settings["spectrogram_length"],
        model_settings["dct_coefficient_count"],
        1,
    ]
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]
    filters = 16
    weight_decay = 1e-4
    # regularizer = l2(weight_decay)
    regularizer = keras.regularizers.l2(l2=weight_decay)
    final_pool_size = (int(input_shape[0] / 2), int(input_shape[1] / 2))

    # Model layers
    # Input pure conv2d
    inputs = keras.Input(shape=(model_settings["fingerprint_size"],), name="input")
    x = keras.layers.Reshape((input_time_size, input_frequency_size, 1))(inputs)
    x = keras.layers.Conv2D(
        filters, (10, 4), strides=(2, 2), padding="same", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizer,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), padding="same", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # Second layer of separable depthwise conv2d
    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizer,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), padding="same", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # Third layer of separable depthwise conv2d
    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizer,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), padding="same", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # Fourth layer of separable depthwise conv2d
    x = keras.layers.DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizer,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), padding="same", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # Reduce size and apply final softmax
    x = keras.layers.Dropout(rate=0.4)(x)

    x = keras.layers.AveragePooling2D(pool_size=final_pool_size)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(model_settings["label_count"], activation="softmax")(x)

    # Instantiate model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_model() -> keras.Model:
    model: keras.Model = tiny_ml_model_r2(model_settings)
    print(model.summary())
    return model


def train(model: keras.Model) -> keras.Model:
    # from keras.callbacks import EarlyStopping
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Define early stopping callback
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(
        x=train_data,
        steps_per_epoch=int(constants.eval_step_interval / 1),
        epochs=30,
        validation_data=val_data,
    )
    return model


def evaluate(model: keras.Model) -> keras.Model:
    logging.info("Evaluating...")
    x_test = np.concatenate([x for x, y in test_data])
    y_test = np.concatenate([y for x, y in test_data])
    # expected_indices = np.concatenate([y for x, y in test_data])
    score = model.evaluate(x=x_test, y=y_test)
    logging.warning(score)


def predict(model: keras.Model) -> keras.Model:
    logging.info("Predicting...")

    def calculate_accuracy(predicted_indices, expected_indices):
        """Calculates and returns accuracy.

        Args:
            predicted_indices: List of predicted integer indices.
            expected_indices: List of expected integer indices.

        Returns:
            Accuracy value between 0 and 1.
        """
        logging.info(f"Predicted indices: {predicted_indices}")
        logging.info(f"Expected indices: {expected_indices}")
        correct_prediction = tf.equal(predicted_indices, expected_indices)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    # VAL DATA
    # expected_indices = np.concatenate([y for x, y in val_data])
    # predictions = model.predict(val_data)
    # predicted_indices = tf.argmax(predictions, axis=1)
    # val_accuracy = calculate_accuracy(predicted_indices, expected_indices)
    # confusion_matrix = tf.math.confusion_matrix(
    #     expected_indices, predicted_indices, num_classes=model_settings["label_count"]
    # )
    # print(confusion_matrix.numpy())
    # print(
    #     f"Validation accuracy = {val_accuracy * 100:.2f}%"
    #     f"(N={audio_processor.set_size(audio_processor.Modes.VALIDATION)})"
    # )
    # TEST DATA
    expected_indices = np.concatenate([y for x, y in test_data])
    predictions = model.predict(test_data)
    predicted_indices = tf.argmax(predictions, axis=1)
    test_accuracy = calculate_accuracy(predicted_indices, expected_indices)
    confusion_matrix = tf.math.confusion_matrix(
        expected_indices, predicted_indices, num_classes=model_settings["label_count"]
    )
    print(confusion_matrix.numpy())
    print(
        f"Test accuracy = {test_accuracy * 100:.2f}%"
        f"(N={audio_processor.set_size(audio_processor.Modes.TESTING)})"
    )

    logging.info("Showing the first 10 predictions")
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Predicted", "Original", "Confidence", "Match"]
    # x_test = np.concatenate([x for x, y in test_data])
    y_test = np.concatenate([y for x, y in test_data])
    y_test = keras.utils.to_categorical(y_test, num_classes=model_settings["label_count"])
    # predictions = model.predict(test_data)
    for i in range(10):
        print(predictions[i])
        table.add_row(
            [
                np.argmax(predictions[i]),
                np.argmax(y_test[i]),
                np.max(predictions[i]),
                "✅" if np.argmax(predictions[i]) == np.argmax(y_test[i]) else "❌",
            ]
        )
    print(table)


def save_model(model: keras.Model, format: str = "keras", model_name: str = ""):
    models_dir = pathlib.Path(MODEL_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    # Save in all formats
    match format:
        case "h5":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.h5", save_format="h5")
        case "tf":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.tf", save_format="tf")
        case "keras":
            model.save(
                f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.keras", save_format="keras"
            )
        case "all":  # save all
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.h5", save_format="h5")
            model.save(
                f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.keras", save_format="keras"
            )
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.tf", save_format="tf")


def load_model(path: str = f"{MODEL_DIR}/{NET_TYPE}_model.keras") -> keras.Model:
    model = keras.models.load_model(path)
    return model


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI utility that enables training and evaluation. The model is always saved."
    )
    parser.add_argument("-l", "--load", help="Load a saved model")
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train the model (creates one or use the loaded one)",
    )
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Evaluate the model and show the first 10 predictions.",
    )
    parser.add_argument(
        "--lite",
        help="Convert to TFLite and evaluate it. Available modes:[dyn,float16,int]",
    )
    parser.add_argument("--summary", action="store_true", help="Show the model summary")
    parser.add_argument("--name", help="Model name (default: " ")", default="")
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
        tf_utils = TFLiteUtils(model, args.lite, model_name=NET_TYPE + args.name)
        # BUG too much time to concatenate train set
        # x_train = np.concatenate([x for x, y in train_data])
        # y_train = np.concatenate([y for x, y in train_data])
        x_test = np.concatenate([x for x, y in test_data])
        y_test = np.concatenate([y for x, y in test_data])
        tf_utils.set_dataset(
            x_train=None,
            y_train=None,
            x_test=x_test,
            y_test=keras.utils.to_categorical(
                y_test, num_classes=model_settings["label_count"]
            ),
        )
        tf_utils.convert_to_tflite(repr_dataset=SpeechCommand.representative_data_gen)
    if args.summary:
        model.summary()
        # convert_to_tflite(model, args.lite, model_name="")
    save_model(model, model_name=NET_TYPE + args.name)


if __name__ == "__main__":
    main()
