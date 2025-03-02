import tensorflow as tf
import numpy as np
from legacy import MaxPool2D

class TestMaxPool(tf.test.TestCase):
    def test_maxpool(self):
        input_shape = (1, 4, 4, 1)
        output_shape = (1, 2, 2, 1)
        maxpool = MaxPool2D(input_shape, output_shape)
        maxpool.set_filter_size(2, 2)
        maxpool.set_stride(2, 2)
        maxpool.set_padding("VALID")
        input = np.array([[[[1], [2], [3], [4]],
                           [[5], [6], [7], [8]],
                           [[9], [10], [11], [12]],
                           [[13], [14], [15], [16]]]], dtype=np.int8)
        output = maxpool.max_pool2d(input)
        expected_output = np.array([[[[6], [8]],
                                     [[14], [16]]]], dtype=np.int8)
        tf_out = tf.nn.max_pool2d(input, ksize=2, strides=2, padding="VALID")
        self.assertAllEqual(output, tf_out.numpy())
        self.assertAllEqual(output, expected_output)

    def test_maxpool_same(self):
        input_shape = (1, 4, 4, 1)
        output_shape = (1, 2, 2, 1)
        maxpool = MaxPool2D(input_shape, output_shape)
        maxpool.set_filter_size(2, 2)
        maxpool.set_stride(2, 2)
        maxpool.set_padding("SAME")
        input = np.array([[[[1], [2], [3], [4]],
                           [[5], [6], [7], [8]],
                           [[9], [10], [11], [12]],
                           [[13], [14], [15], [16]]]], dtype=np.int8)
        output = maxpool.max_pool2d(input)
        expected_output = np.array([[[[6], [8]],
                                     [[14], [16]]]], dtype=np.int8)
        tf_out = tf.nn.max_pool2d(input, ksize=2, strides=2, padding="SAME")
        self.assertAllEqual(output, tf_out.numpy())
        self.assertAllEqual(output, expected_output)

    def test_maxpool_same_odd(self):
        input_shape = (1, 5, 5, 1)
        output_shape = (1, 3, 3, 1)
        maxpool = MaxPool2D(input_shape, output_shape)
        maxpool.set_filter_size(3, 3)
        maxpool.set_stride(2, 2)
        maxpool.set_padding("SAME")
        input = np.array([[[[1], [2], [3], [4], [5]],
                           [[6], [7], [8], [9], [10]],
                           [[11], [12], [13], [14], [15]],
                           [[16], [17], [18], [19], [20]],
                           [[21], [22], [23], [24], [25]]]], dtype=np.int8)
        output = maxpool.max_pool2d(input)
        tf_out = tf.nn.max_pool2d(input, ksize=3, strides=2, padding="SAME")
        self.assertAllEqual(output, tf_out.numpy())

    def test_maxpool_same_odd_2(self):
        input_shape = (1, 5, 5, 1)
        output_shape = (1, 3, 3, 1)
        maxpool = MaxPool2D(input_shape, output_shape)
        maxpool.set_filter_size(2, 2)
        maxpool.set_stride(2, 2)
        maxpool.set_padding("SAME")
        input = np.array([[[[1], [2], [3], [4], [5]],
                           [[6], [7], [8], [9], [10]],
                           [[11], [12], [13], [14], [15]],
                           [[16], [17], [18], [19], [20]],
                           [[21], [22], [23], [24], [25]]]], dtype=np.int8)
        output = maxpool.max_pool2d(input)
        tf_out = tf.nn.max_pool2d(input, ksize=2, strides=2, padding="SAME")
        self.assertAllEqual(output, tf_out.numpy())

    def test_maxpool_generate_code(self):
        input_shape = (1, 5, 5, 1)
        output_shape = (1, 3, 3, 1)
        maxpool = MaxPool2D(input_shape, output_shape)
        maxpool.set_filter_size(2, 2)
        maxpool.set_stride(2, 2)
        maxpool.set_padding("VALID")
        input = np.array([[[[1], [2], [3], [4], [5]],
                           [[6], [7], [8], [9], [10]],
                           [[11], [12], [13], [14], [15]],
                           [[16], [17], [18], [19], [20]],
                           [[21], [22], [23], [24], [25]]]], dtype=np.int8)
        output = maxpool.max_pool2d(input)
        print(output)
        code = maxpool.generate_code_v2()
        code += "\n"
        code += maxpool.generate_code_v3()
        with open("test_generated/src/maxpool.c", "w") as f:
            f.write(code)

    def test_maxpool_with_padding(self):
        input_shape = (1, 5, 5, 1)
        output_shape = (1, 2, 2, 1)
        maxpool = MaxPool2D(input_shape, output_shape)
        maxpool.set_filter_size(3, 3)
        maxpool.set_stride(2, 2)
        maxpool.set_padding("VALID")
        input = np.array([[[[1], [2], [3], [4], [5]],
                           [[6], [7], [8], [9], [10]],
                           [[11], [12], [13], [14], [15]],
                           [[16], [17], [18], [19], [20]],
                           [[21], [22], [23], [24], [25]]]], dtype=np.int8)
        output = maxpool.max_pool2d(input)
        expected_output = np.array([[[[13], [15]],
                                     [[23], [25]]]], dtype=np.int8)
        tf_out = tf.nn.max_pool2d(input, ksize=3, strides=2, padding="VALID")
        self.assertAllEqual(output, tf_out.numpy())
        self.assertAllEqual(output, expected_output)

# Running the tests
if __name__ == '__main__':
    tf.test.main()