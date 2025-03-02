'''
Things to remember:
- The filter is a 4D tensor with shape [filter_height, filter_width, in_channels, out_channels]
- The input is a 4D tensor with shape [batch, in_height, in_width, in_channels]
- The output is a 4D tensor with shape [batch, out_height, out_width, out_channels]
- The stride is a list of 4 ints: [1, stride, stride, 1]
- The padding is either 'VALID' or 'SAME'
- tensorflow wants filters with shape [filter_height, filter_width, in_channels, out_channels]
- we use filters with shape [out_channels, filter_height, filter_width, in_channels]
'''
import tensorflow as tf
import numpy as np
import unittest
from legacy import Conv2D

def compute_padding(output_size, input_size, filter_size, stride):
        pad = ((output_size - 1) * stride + filter_size - input_size)
        offset = pad % 2
        if pad < 0:
            return 0
        return (pad // 2, offset)

# Function to manually perform convolution operation
def manual_convolve(input, filters, output_shape, padding="VALID", stride=(1, 1)):
    filters = tf.transpose(filters, perm=[3, 0, 1, 2])

    stride_height = stride[0]
    stride_width = stride[1]
    if padding == "VALID":
        padding_height = 0
        padding_width = 0
    elif padding == "SAME":
        padding_height, _ = compute_padding(output_shape[1], input.shape[1], filters.shape[1], stride_height)
        padding_width, _ = compute_padding(output_shape[2], input.shape[2], filters.shape[2], stride_width)

    if input.shape[0] != output_shape[0]:
        raise ValueError("Batch size mismatch")
    batches = input.shape[0]

    if filters.shape[0] != output_shape[3]:
        raise ValueError("Output depth mismatch", filters.shape[0], output_shape[3])
    output_depth = filters.shape[0]

    input_depth = input.shape[3]

    input_height = input.shape[1]
    input_width = input.shape[2]
    filter_height = filters.shape[1]
    filter_width = filters.shape[2]
    filter_input_depth = filters.shape[3]

    output_height = output_shape[1]
    output_width = output_shape[2]

    output = np.zeros(output_shape)

    for batch in range(batches):
        for out_h in range(output_height):
            in_y_origin = (out_h * stride_height) - padding_height
            for out_w in range(output_width):
                in_x_origin = (out_w * stride_width) - padding_width
                for out_channel in range(output_depth):
                    accumulator = 0
                    for filter_h in range(filter_height):
                        in_y = in_y_origin + filter_h
                        for filter_w in range(filter_width):
                            in_x = in_x_origin + filter_w
                            is_point_valid = in_y >= 0 and in_y < input_height and in_x >= 0 and in_x < input_width
                            if is_point_valid:
                                for in_channel in range(filter_input_depth):
                                    input_val = input[batch, in_y, in_x, in_channel]
                                    filter_val = filters[out_channel, filter_h, filter_w, in_channel] 
                                    accumulator += input_val * filter_val
                    output[batch, out_h, out_w, out_channel] = accumulator

    
    return tf.convert_to_tensor(output, dtype=tf.float32)


# Function to perform convolution operation
def convolve(image, filters, padding, strides=[1, 1, 1, 1]):
    conv = tf.nn.conv2d(image, filters, strides=strides, padding=padding)
    return conv


# Unit test class using TensorFlow's testing framework
class TestConvolution(tf.test.TestCase):
    def setUp(self):
        super(TestConvolution, self).setUp()
        # Create a 28x28 test image with 1 channel (grayscale)
        self.image = tf.random.uniform(shape=[1, 28, 28, 1], dtype=tf.float32)

    # @unittest.skip("Skipping one filter test")
    def test_one_filter(self):
        # Create a 3x3 filter with 1 input channel and 1 output channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "VALID"
        stride = (1,1)
        legacy_conv = Conv2D(self.image.shape, [1, 26, 26, 1])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])
        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 26, 26, 1], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding) #convolve(self.image, filter, padding='VALID')
        leg_conv = legacy_conv.conv2d(self.image)
        #print("Manual: ", manual)
        #print("TF: ", tf_conv)
        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)
        
        # # Check the shape of the output (should be [1, 26, 26, 1] for no padding)
        # self.assertEqual(result.shape, (1, 26, 26, 1))

    # @unittest.skip("Skipping two filters test")
    def test_two_filters(self):
        # Create a 3x3 filter with 1 input channel and 2 output channels
        filters = tf.random.uniform(shape=[3, 3, 1, 2], dtype=tf.float32)
        padding = "VALID"
        stride = (1,1)
        legacy_conv = Conv2D(self.image.shape, [1, 26, 26, 2])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])
        
        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 26, 26, 2], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding) #convolve(self.image, filter, padding='VALID')
        leg_conv = legacy_conv.conv2d(self.image)

        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)

    def test_one_filter_with_padding(self):
        # Create a 3x3 filter with 1 input channel and 1 output channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "SAME"
        stride = (1,1)
        legacy_conv = Conv2D(self.image.shape, [1, 28, 28, 1])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])
        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 28, 28, 1], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding) #convolve(self.image, filter, padding='VALID')
        leg_conv = legacy_conv.conv2d(self.image)
        
        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)

    def test_two_filters_with_padding(self):
        # Create a 3x3 filter with 1 input channel and 2 output channels
        filters = tf.random.uniform(shape=[3, 3, 1, 2], dtype=tf.float32)
        padding = "SAME"
        stride = (1,1)
        legacy_conv = Conv2D(self.image.shape, [1, 28, 28, 2])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])
        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 28, 28, 2], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding) #convolve(self.image, filter, padding='VALID')
        leg_conv = legacy_conv.conv2d(self.image)

        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)
        
    
    def test_one_filter_with_strides(self):
        # Create a 3x3 filter with 1 input channel and 1 output channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "VALID"
        stride = (2,2)
        legacy_conv = Conv2D(self.image.shape, [1, 13, 13, 1])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])
        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 13, 13, 1], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding) #convolve(self.image, filter, padding='VALID')
        leg_conv = legacy_conv.conv2d(self.image)
        
        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)

    def test_two_filters_with_strides(self):
        # Create a 3x3 filter with 1 input channel and 2 output channels
        filters = tf.random.uniform(shape=[3, 3, 1, 2], dtype=tf.float32)
        padding = "VALID"
        stride = (2,2)
        legacy_conv = Conv2D(self.image.shape, [1, 13, 13, 2])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])
        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 13, 13, 2], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding) #convolve(self.image, filter, padding='VALID')
        leg_conv = legacy_conv.conv2d(self.image)
        
        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)
    
    def test_one_filter_with_padding_and_strides(self):
        # Create a 3x3 filter with 1 input channel and 1 output channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "SAME"
        stride = (2,2)
        legacy_conv = Conv2D(self.image.shape, [1, 14, 14, 1])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])
        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 14, 14, 1], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding) #convolve(self.image, filter, padding='VALID')
        leg_conv = legacy_conv.conv2d(self.image)
        
        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)
    
    def test_two_filters_with_padding_and_strides(self):
        # Create a 3x3 filter with 1 input channel and 2 output channels
        filters = tf.random.uniform(shape=[3, 3, 1, 2], dtype=tf.float32)
        padding = "SAME"
        stride = (2,2)
        legacy_conv = Conv2D(self.image.shape, [1, 14, 14, 2])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])

        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 14, 14, 2], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding) #convolve(self.image, filter, padding='VALID')
        leg_conv = legacy_conv.conv2d(self.image)

        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)
    
    def test_one_filter_with_padding_and_strides_unequal(self):
        # Create a 3x3 filter with 1 input channel and 1 output channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "SAME"
        stride = (3,2)
        legacy_conv = Conv2D(self.image.shape, [1, 10, 14, 1])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])

        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 10, 14, 1], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding) #convolve(self.image, filter, padding='VALID')
        leg_conv = legacy_conv.conv2d(self.image)
        
        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)
    
    def test_two_filters_with_padding_and_strides_unequal(self):
        # Create a 3x3 filter with 1 input channel and 2 output channels
        filters = tf.random.uniform(shape=[3, 3, 1, 2], dtype=tf.float32)
        padding = "SAME"
        stride = (3,2)
        legacy_conv = Conv2D(self.image.shape, [1, 10, 14, 2])
        legacy_conv.set_filters(filters)
        legacy_conv.set_padding(padding)
        legacy_conv.set_strides(stride[0], stride[1])

        # Perform the convolution
        manual = manual_convolve(self.image, filters, output_shape=[1, 10, 14, 2], padding=padding, stride=stride)
        tf_conv = tf.nn.conv2d(self.image, filters, strides=[1, stride[0], stride[1], 1], padding=padding)
        leg_conv = legacy_conv.conv2d(self.image)
        
        self.assertAllClose(manual, tf_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(manual, leg_conv, rtol=1e-5, atol=1e-5)

# Running the tests
if __name__ == '__main__':
    tf.test.main()
