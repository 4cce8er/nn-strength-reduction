import tensorflow as tf
import keras.datasets.mnist as mnist
from logger.logger import logging
import keras
from . import Dataset

class MNIST_DS(Dataset):
    def __init__(self):
        mnist_train, mnist_test = mnist.load_data()
        self.train_data, self.train_labels = mnist_train
        self.test_data, self.test_labels = mnist_test
        self.__preprocess__()

    def __preprocess__(self):
        logging.info("Initial shape: {}".format(self.train_data[0].shape))
        # Scale the images to the [0,1] range
        self.train_data_scaled = tf.cast(self.train_data, dtype=tf.float32) / 255.0
        self.test_data_scaled = tf.cast(self.test_data, dtype=tf.float32) / 255.0
        # Now we have the labels in the form [5 0 4 ... 5 6 8]. We want that each one is represented as a vector (10,1)
        self.train_labels = keras.utils.to_categorical(self.train_labels, num_classes=10)
        self.test_labels = keras.utils.to_categorical(self.test_labels, num_classes=10)
        assert self.train_labels[0].shape == (10,)

    def get_data(self):
        '''
        Returns the data as is, values between 0 and 255
        '''
        return self.train_data, self.train_labels, self.test_data, self.test_labels
    
    def get_scaled_data(self):
        '''
        Returns the scaled data between 0 and 1
        '''
        return self.train_data_scaled, self.train_labels, self.test_data_scaled, self.test_labels

    def representative_data_gen(self):
        '''
        Returns a generator of representative data for the quantization process
        '''
        small_dataset = tf.data.Dataset.from_tensor_slices(self.train_data_scaled).batch(1)
        for input_value in small_dataset.take(100):
            yield [input_value]

    def expand_dims(self) -> Dataset:
        '''
        Expands the dimensions of the data
        '''
        self.train_data_scaled = tf.expand_dims(self.train_data_scaled, axis=-1)
        self.test_data_scaled = tf.expand_dims(self.test_data_scaled, axis=-1)
        self.train_data = tf.expand_dims(self.train_data, axis=-1)
        self.test_data = tf.expand_dims(self.test_data, axis=-1)
        logging.info(self.train_data_scaled.shape)
        return self

Mnist = MNIST_DS()