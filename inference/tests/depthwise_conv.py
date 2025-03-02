import tensorflow as tf
import numpy as np
import unittest
from legacy import DepthwiseConv2D


def compute_padding(output_size, input_size, filter_size, stride):
    pad = (output_size - 1) * stride + filter_size - input_size
    offset = pad % 2
    if pad < 0:
        return 0
    return (pad // 2, offset)


def depthwise_convolve(image, filters, padding, strides=[1, 1, 1, 1]):
    conv = tf.nn.depthwise_conv2d(image, filters, strides=strides, padding=padding)
    return conv


def manual_depthwise_convolve(input, filters, output_shape, padding, strides):
    filters = tf.transpose(filters, perm=[3, 0, 1, 2])

    stride_height = strides[0]
    stride_width = strides[1]
    if padding == "VALID":
        padding_height = 0
        padding_width = 0
    elif padding == "SAME":
        padding_height, _ = compute_padding(
            output_shape[1], input.shape[1], filters.shape[1], stride_height
        )
        padding_width, _ = compute_padding(
            output_shape[2], input.shape[2], filters.shape[2], stride_width
        )

    if input.shape[0] != output_shape[0]:
        raise ValueError("Batch size mismatch")
    batches = input.shape[0]
    if filters.shape[3] != output_shape[3]:
        raise ValueError(
            "Output depth mismatch",
            filters.shape[3],
            output_shape[3],
            filters.shape,
            output_shape,
        )
    output_depth = filters.shape[3]
    input_height = input.shape[1]
    input_width = input.shape[2]
    input_depth = input.shape[3]
    filter_height = filters.shape[1]
    filter_width = filters.shape[2]
    output_height = output_shape[1]
    output_width = output_shape[2]
    output = np.zeros(output_shape)
    print(batches)
    for batch in range(batches):
        for out_h in range(output_height):
            for out_w in range(output_width):
                for in_channel in range(input_depth):
                    output_channel = in_channel
                    in_x_origin = (out_w * stride_width) - padding_width
                    in_y_origin = (out_h * stride_height) - padding_height
                    accumulator = 0
                    for filter_h in range(filter_height):
                        for filter_w in range(filter_width):
                            in_x = in_x_origin + filter_w
                            in_y = in_y_origin + filter_h
                            is_point_valid = (
                                in_x >= 0
                                and in_x < input_width
                                and in_y >= 0
                                and in_y < input_height
                            )
                            if is_point_valid:
                                input_val = input[batch, in_y, in_x, in_channel]
                                filter_val = filters[0, filter_h, filter_w, in_channel]
                                accumulator += input_val * filter_val
                    output[batch, out_h, out_w, output_channel] = accumulator

    return tf.convert_to_tensor(output, dtype=tf.float32)


def manual_depthwise_convolve_2(input, filters, output_shape, padding, strides):
    filters = tf.transpose(filters, perm=[3, 0, 1, 2])

    output = np.zeros(output_shape)

    for out_h in range(output_shape[1]):
        for out_w in range(output_shape[2]):
            for in_channel in range(input.shape[3]):
                accumulator = 0
                for filter_h in range(filters.shape[1]):
                    for filter_w in range(filters.shape[2]):
                        accumulator += (
                            input[0, out_h + filter_h, out_w + filter_w, in_channel]
                            * filters[0, filter_h, filter_w, in_channel]
                        )
                output[0, out_h, out_w, in_channel] = accumulator

    return tf.convert_to_tensor(output, dtype=tf.float32)


class TestDepthwiseConv(tf.test.TestCase):
    def setUp(self):
        super(TestDepthwiseConv, self).setUp()
        # Create a 28x28 test image with 1 channel (grayscale)
        self.image = tf.random.uniform(shape=[1, 28, 28, 1], dtype=tf.float32)

    def test_one_filter(self):
        # Create a 3x3 depthwise filter with 1 channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "VALID"
        strides = [1, 1, 1, 1]
        dw_legacy = DepthwiseConv2D(self.image.shape, [1, 26, 26, 1])
        dw_legacy.set_filters(filters.numpy())
        dw_legacy.set_padding(padding)
        dw_legacy.set_strides(1, 1)
        conv_legacy = dw_legacy.depthwise_conv2d(self.image.numpy())
        conv = depthwise_convolve(self.image, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.image.numpy(), filters.numpy(), conv.shape, padding, (1, 1)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(conv, conv_legacy, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 26, 26, 1])

    def test_two_filters(self):
        self.inputs = tf.random.uniform(shape=[1, 28, 28, 2], dtype=tf.float32)
        # Create a 3x3 depthwise filter with 2 channels
        filters = tf.random.uniform(shape=[3, 3, 2, 1], dtype=tf.float32)
        padding = "VALID"
        strides = [1, 1, 1, 1]
        dw_legacy = DepthwiseConv2D(self.inputs.shape, [1, 26, 26, 2])
        dw_legacy.set_filters(filters.numpy())
        dw_legacy.set_padding(padding)
        dw_legacy.set_strides(1, 1)
        conv_legacy = dw_legacy.depthwise_conv2d(self.inputs.numpy())
        conv = depthwise_convolve(self.inputs, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.inputs.numpy(), filters.numpy(), conv.shape, padding, (1, 1)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(conv, conv_legacy, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 26, 26, 2])

    def test_one_filter_with_padding(self):
        # Create a 3x3 depthwise filter with 1 channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "SAME"
        strides = [1, 1, 1, 1]
        dw_legacy = DepthwiseConv2D(self.image.shape, [1, 28, 28, 1])
        dw_legacy.set_filters(filters.numpy())
        dw_legacy.set_padding(padding)
        dw_legacy.set_strides(1, 1)
        conv_legacy = dw_legacy.depthwise_conv2d(self.image.numpy())
        conv = depthwise_convolve(self.image, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.image.numpy(), filters.numpy(), conv.shape, padding, (1, 1)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(conv, conv_legacy, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 28, 28, 1])

    def test_two_filters_with_padding(self):
        self.inputs = tf.random.uniform(shape=[1, 28, 28, 2], dtype=tf.float32)
        # Create a 3x3 depthwise filter with 2 channels
        filters = tf.random.uniform(shape=[3, 3, 2, 1], dtype=tf.float32)
        padding = "SAME"
        strides = [1, 1, 1, 1]
        dw_legacy = DepthwiseConv2D(self.inputs.shape, [1, 28, 28, 2])
        dw_legacy.set_filters(filters.numpy())
        dw_legacy.set_padding(padding)
        dw_legacy.set_strides(1, 1)
        conv_legacy = dw_legacy.depthwise_conv2d(self.inputs.numpy())
        conv = depthwise_convolve(self.inputs, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.inputs.numpy(), filters.numpy(), conv.shape, padding, (1, 1)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(conv, conv_legacy, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 28, 28, 2])

    def test_one_filter_with_strides(self):
        # Create a 3x3 depthwise filter with 1 channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "VALID"
        strides = [1, 2, 2, 1]
        dw_legacy = DepthwiseConv2D(self.image.shape, [1, 13, 13, 1])
        dw_legacy.set_filters(filters.numpy())
        dw_legacy.set_padding(padding)
        dw_legacy.set_strides(2, 2)
        conv_legacy = dw_legacy.depthwise_conv2d(self.image.numpy())
        conv = depthwise_convolve(self.image, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.image.numpy(), filters.numpy(), conv.shape, padding, (2, 2)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(conv, conv_legacy, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 13, 13, 1])

    def test_two_filters_with_strides(self):
        self.inputs = tf.random.uniform(shape=[1, 28, 28, 2], dtype=tf.float32)
        # Create a 3x3 depthwise filter with 2 channels
        filters = tf.random.uniform(shape=[3, 3, 2, 1], dtype=tf.float32)
        padding = "VALID"
        strides = [1, 2, 2, 1]
        dw_legacy = DepthwiseConv2D(self.inputs.shape, [1, 13, 13, 2])
        dw_legacy.set_filters(filters.numpy())
        dw_legacy.set_padding(padding)
        dw_legacy.set_strides(2, 2)
        conv_legacy = dw_legacy.depthwise_conv2d(self.inputs.numpy())
        conv = depthwise_convolve(self.inputs, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.inputs.numpy(), filters.numpy(), conv.shape, padding, (2, 2)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(conv, conv_legacy, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 13, 13, 2])

    def test_one_filter_with_padding_and_strides(self):
        # Create a 3x3 depthwise filter with 1 channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "SAME"
        strides = [1, 2, 2, 1]
        dw_legacy = DepthwiseConv2D(self.image.shape, [1, 14, 14, 1])
        dw_legacy.set_filters(filters.numpy())
        dw_legacy.set_padding(padding)
        dw_legacy.set_strides(2, 2)
        conv_legacy = dw_legacy.depthwise_conv2d(self.image.numpy())
        conv = depthwise_convolve(self.image, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.image.numpy(), filters.numpy(), conv.shape, padding, (2, 2)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)

        self.assertAllClose(conv, conv_legacy, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 14, 14, 1])

    def test_two_filters_with_padding_and_strides(self):
        self.inputs = tf.random.uniform(shape=[1, 28, 28, 2], dtype=tf.float32)
        # Create a 3x3 depthwise filter with 2 channels
        filters = tf.random.uniform(shape=[3, 3, 2, 1], dtype=tf.float32)
        padding = "SAME"
        strides = [1, 2, 2, 1]
        dw_legacy = DepthwiseConv2D(self.inputs.shape, [1, 14, 14, 2])
        dw_legacy.set_filters(filters.numpy())
        dw_legacy.set_padding(padding)
        dw_legacy.set_strides(2, 2)
        conv_legacy = dw_legacy.depthwise_conv2d(self.inputs.numpy())
        conv = depthwise_convolve(self.inputs, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.inputs.numpy(), filters.numpy(), conv.shape, padding, (2, 2)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(conv, conv_legacy, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 14, 14, 2])

    @unittest.skip("Not implemented in tensorflow")
    def test_one_filter_with_padding_and_strides_unequal(self):
        # Create a 3x3 depthwise filter with 1 channel
        filters = tf.random.uniform(shape=[3, 3, 1, 1], dtype=tf.float32)
        padding = "SAME"
        strides = [1, 2, 3, 1]
        dw_legacy = DepthwiseConv2D(self.image.shape, [1, 10, 10, 1])
        dw_legacy.set_filters(filters.numpy())
        dw_legacy.set_padding(padding)
        dw_legacy.set_strides(2, 3)
        conv_legacy = dw_legacy.depthwise_conv2d(self.image.numpy())
        conv = depthwise_convolve(self.image, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.image.numpy(), filters.numpy(), conv.shape, padding, (2, 3)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)
        self.assertAllClose(conv, conv_legacy, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 10, 10, 1])

    @unittest.skip("Not implemented in tensorflow")
    def test_two_filters_with_padding_and_strides_unequal(self):
        self.inputs = tf.random.uniform(shape=[1, 28, 28, 2], dtype=tf.float32)
        # Create a 3x3 depthwise filter with 2 channels
        filters = tf.random.uniform(shape=[3, 3, 2, 1], dtype=tf.float32)
        padding = "SAME"
        strides = [1, 2, 3, 1]
        conv = depthwise_convolve(self.inputs, filters, padding, strides)
        manual_conv = manual_depthwise_convolve(
            self.inputs.numpy(), filters.numpy(), conv.shape, padding, (2, 3)
        )

        self.assertAllClose(conv, manual_conv, rtol=1e-5, atol=1e-5)
        self.assertEqual(conv.shape, [1, 10, 10, 2])

    def test_real_dat(self):
        values = [
            -25,
            70,
            -23,
            -13,
            -47,
            127,
            -54,
            108,
            -87,
            77,
            58,
            43,
            114,
            123,
            99,
            -58,
            -4,
            66,
            -44,
            120,
            -101,
            104,
            103,
            13,
            -10,
            -38,
            -82,
            -29,
            -77,
            31,
            73,
            26,
            -26,
            -78,
            -115,
            55,
            -79,
            -40,
            35,
            -38,
            -91,
            -35,
            -123,
            -105,
            -40,
            105,
            -34,
            84,
            43,
            50,
            77,
            70,
            27,
            52,
            -44,
            -111,
            -114,
            2,
            -12,
            -63,
            -95,
            -67,
            92,
            7,
            -16,
            105,
            -66,
            33,
            -63,
            97,
            124,
            -25,
            -66,
            -127,
            -2,
            23,
            106,
            92,
            -21,
            22,
            15,
            -72,
            -36,
            -86,
            108,
            48,
            -69,
            123,
            -78,
            47,
            -68,
            -44,
            108,
            -104,
            91,
            -36,
            -126,
            -102,
            126,
            -61,
            123,
            122,
            42,
            -70,
            123,
            -87,
            81,
            102,
            -123,
            -68,
            -4,
            20,
            -11,
            88,
            62,
            -31,
            9,
            121,
            -36,
            59,
            40,
            25,
            -113,
            21,
            49,
            107,
            113,
            51,
            -123,
            111,
            119,
            -128,
            105,
            33,
            -70,
            101,
            74,
            -117,
            75,
            80,
            -56,
            -57,
            -28,
            61,
            -97,
            -93,
            -98,
            40,
            -100,
            -5,
            -28,
            69,
            -108,
            -13,
            -38,
            69,
            -34,
            -53,
            -7,
            -29,
            -69,
            -16,
            -28,
            -92,
            -111,
            30,
            -119,
            92,
            42,
            84,
            44,
            114,
            -101,
            -112,
            47,
            -69,
            -77,
            77,
            99,
            -48,
            -56,
            -57,
            -107,
            -36,
            59,
            -17,
            -94,
            -103,
            58,
            27,
            -3,
            117,
            -117,
            97,
            -102,
            -100,
            -1,
            -93,
            120,
            -87,
            120,
            36,
            -101,
            -109,
            53,
            74,
            -50,
            104,
            24,
            -78,
            -72,
            96,
            -7,
            -51,
            -67,
            -76,
            60,
            -33,
            -50,
            -9,
            122,
            75,
            -20,
            -123,
            44,
            6,
            -95,
            -85,
            42,
            -102,
            -43,
            34,
            62,
            -16,
            53,
            -13,
            -69,
            -124,
            -36,
            83,
            -74,
            20,
            51,
            47,
            98,
            112,
            100,
            30,
            -49,
            -78,
            -107,
            -55,
            125,
            2,
            -50,
            41,
            -120,
            -16,
            84,
            50,
            10,
            -87,
            -44,
            -56,
            26,
            -118,
            60,
            85,
            -114,
            -104,
            40,
            -60,
            44,
            -37,
            115,
            14,
            -52,
            87,
            -83,
            27,
            -119,
            -62,
            101,
            -122,
            68,
            -77,
            47,
            77,
            35,
            4,
            -1,
            -83,
            45,
            84,
            -10,
            -57,
            94,
            -78,
            -100,
            108,
            -54,
            68,
            -80,
            118,
            -96,
            -93,
            5,
            -20,
            123,
            50,
            -121,
            -124,
            116,
            108,
            -117,
            57,
            -96,
            58,
            6,
            67,
            -66,
            -123,
            113,
            108,
            89,
            -25,
            -77,
            55,
            25,
            -48,
            35,
            99,
            -108,
            83,
            89,
            -76,
            119,
            -34,
            32,
            114,
            -112,
            40,
            118,
            -123,
            20,
            -127,
            62,
            52,
            60,
            -60,
            -8,
            122,
            -55,
            -23,
            102,
            -93,
            80,
            -102,
            90,
            -23,
            -22,
            -2,
            -52,
            -2,
            -47,
            -91,
            51,
            -56,
            4,
            -45,
            -70,
            20,
            123,
            -79,
            25,
            16,
            -78,
            -41,
            -60,
            110,
            27,
            60,
            105,
            101,
            -91,
            79,
            -120,
            117,
            105,
            98,
            -34,
            -45,
            -32,
            42,
            82,
            50,
            80,
            5,
            122,
            -44,
            88,
            -75,
            104,
            84,
            -26,
            2,
            -28,
            24,
            89,
            40,
            7,
            -11,
            -27,
            -16,
            -38,
            10,
            -65,
            -30,
            0,
            -87,
            -60,
            94,
            -4,
            37,
            9,
            -50,
            -41,
            -39,
            83,
            -47,
            45,
            44,
            6,
            21,
            0,
            108,
            -105,
            100,
            5,
            113,
            12,
            -116,
            -26,
            113,
            -4,
            64,
            -4,
            59,
            -94,
            124,
            100,
            -26,
            90,
            -31,
            -117,
            -29,
            47,
            -30,
            60,
            3,
            52,
            -23,
            -81,
            -70,
            127,
            47,
            -89,
            -106,
            19,
            44,
            -121,
            -97,
            56,
            -19,
            -111,
            -76,
            -83,
            13,
            111,
            -49,
            9,
            84,
            54,
            -29,
            -75,
            65,
            71,
            100,
            -92,
            3,
        ]
        self.inputs = tf.convert_to_tensor(
            np.array(values).reshape(1, 49, 10, 1), dtype=tf.int32
        )
        # Conv filters
        filter_values = [
            [[[63], [4], [-43], [-26]], [[17], [5], [-16], [-52]], [[32], [42], [9], [-51]], [[7], [83], [57], [-5]], [[-81], [39], [79], [80]], [[-99], [-89], [18], [127]], [[-19], [-86], [-28], [63]], [[30], [-17], [-30], [-23]], [[26], [22], [-47], [-45]], [[34], [14], [-25], [-36]]],
            [[[12], [-26], [9], [46]], [[-20], [-72], [14], [89]], [[-7], [-73], [20], [127]], [[27], [0], [-14], [63]], [[13], [42], [-33], [-17]], [[-12], [51], [-19], [-92]], [[-19], [40], [7], [-120]], [[-6], [36], [4], [-71]], [[5], [32], [3], [-52]], [[14], [10], [-5], [-21]]],
            [[[-10], [70], [-40], [42]], [[-17], [127], [-98], [9]], [[92], [104], [-97], [33]], [[79], [3], [21], [53]], [[-8], [-110], [111], [32]], [[-58], [-120], [108], [-27]], [[-35], [-53], [63], [-50]], [[3], [2], [6], [-47]], [[-3], [6], [-43], [-15]], [[-35], [-32], [-72], [11]]],
            [[[-46], [33], [-13], [66]], [[5], [-1], [19], [114]], [[0], [-49], [38], [127]], [[-33], [-56], [14], [77]], [[-39], [-35], [4], [37]], [[-13], [-8], [-13], [14]], [[11], [1], [-16], [-4]], [[-22], [4], [-5], [13]], [[-41], [8], [-1], [12]], [[-5], [-4], [1], [12]]],
            [[[-24], [-86], [127], [51]], [[-16], [-106], [91], [22]], [[24], [-44], [75], [-27]], [[31], [2], [43], [-23]], [[0], [17], [22], [12]], [[-18], [21], [26], [38]], [[-25], [23], [24], [30]], [[-9], [22], [21], [9]], [[7], [11], [-7], [-22]], [[53], [10], [-87], [-58]]],
            [[[2], [-24], [6], [-48]], [[33], [-53], [19], [-23]], [[105], [-28], [26], [-10]], [[127], [25], [3], [4]], [[41], [34], [-26], [19]], [[-33], [37], [-6], [37]], [[-71], [28], [10], [24]], [[-62], [16], [4], [0]], [[-46], [9], [-11], [-19]], [[-34], [21], [-23], [-41]]],
            [[[32], [-35], [87], [16]], [[20], [-58], [7], [32]], [[16], [-66], [-72], [30]], [[18], [-50], [-127], [10]], [[21], [-15], [-108], [-2]], [[30], [11], [-26], [4]], [[1], [16], [29], [35]], [[-29], [10], [45], [56]], [[-26], [22], [23], [31]], [[-29], [73], [9], [-26]]],
            [[[20], [-5], [-102], [-22]], [[-9], [-8], [-39], [60]], [[-80], [-22], [48], [127]], [[-68], [-6], [104], [55]], [[3], [72], [53], [-54]], [[23], [62], [-2], [-45]], [[-4], [24], [-21], [-15]], [[-9], [-1], [-31], [-12]], [[25], [-32], [5], [-29]], [[105], [-44], [56], [-75]]],
            [[[-60], [-59], [-16], [-34]], [[13], [-108], [15], [-38]], [[81], [97], [78], [-3]], [[91], [127], [38], [72]], [[17], [36], [-52], [-17]], [[-65], [-29], [-78], [-47]], [[-105], [-49], [-54], [13]], [[-25], [-46], [-1], [54]], [[14], [-2], [44], [43]], [[48], [36], [-21], [24]]],
            [[[92], [-40], [14], [65]], [[75], [-114], [49], [64]], [[-10], [-98], [119], [8]], [[-86], [30], [106], [-49]], [[-68], [127], [1], [-84]], [[-26], [124], [-70], [-85]], [[-16], [68], [-68], [-76]], [[-13], [17], [-8], [-36]], [[-51], [17], [41], [26]], [[-20], [55], [18], [85]]],
            [[[47], [-13], [-46], [48]], [[29], [35], [-30], [7]], [[50], [4], [-76], [-70]], [[24], [-39], [-127], [-93]], [[-21], [-29], [-71], [-18]], [[-47], [-4], [12], [52]], [[-42], [7], [51], [37]], [[-4], [7], [46], [16]], [[16], [5], [41], [11]], [[-20], [13], [70], [8]]],
            [[[-66], [91], [64], [53]], [[-19], [46], [46], [-13]], [[-19], [36], [34], [-87]], [[-1], [45], [-12], [-127]], [[24], [47], [-48], [-124]], [[38], [35], [-77], [-59]], [[27], [-1], [-60], [-18]], [[40], [-16], [-42], [-4]], [[38], [-23], [-2], [33]], [[-26], [-78], [33], [70]]],
            [[[-105], [-127], [-74], [-53]], [[-82], [-111], [-43], [-5]], [[-42], [-52], [-8], [-8]], [[-12], [6], [10], [-13]], [[18], [49], [3], [-19]], [[9], [67], [1], [-24]], [[15], [78], [24], [-8]], [[34], [72], [26], [9]], [[58], [48], [19], [3]], [[101], [60], [35], [7]]],
            [[[-7], [27], [-31], [-37]], [[-1], [20], [-13], [-67]], [[-20], [12], [-7], [-89]], [[-25], [13], [-2], [-51]], [[-27], [-26], [20], [63]], [[-18], [-49], [39], [127]], [[6], [-33], [37], [91]], [[17], [7], [11], [37]], [[20], [-1], [-12], [13]], [[55], [-34], [-46], [42]]],
            [[[96], [7], [-31], [12]], [[35], [32], [-6], [26]], [[-33], [45], [17], [3]], [[-127], [5], [18], [-6]], [[-82], [-50], [-36], [-20]], [[3], [-36], [-34], [-12]], [[34], [4], [16], [-10]], [[10], [7], [29], [-3]], [[1], [-1], [22], [7]], [[43], [-5], [-5], [-20]]],
            [[[60], [67], [42], [-26]], [[60], [77], [39], [15]], [[81], [53], [67], [8]], [[44], [18], [51], [-17]], [[4], [-6], [29], [-15]], [[-21], [-26], [14], [4]], [[-30], [-48], [-11], [0]], [[-31], [-60], [-26], [3]], [[-53], [-75], [-40], [-7]], [[-127], [-69], [-30], [5]]]
        ]
        # Apply the first conv layer
        filters = tf.convert_to_tensor(filter_values, dtype=tf.int32)
        print(filters.shape)
        filters = tf.transpose(filters, perm=[1, 2, 3, 0])
            
        padding = "VALID"
        strides = [1, 1, 1, 1]
        conv = tf.nn.conv2d(self.inputs, filters, strides=strides, padding=padding)
        print(conv.shape)
        # Now that we have the convolution, we apply and tests the depthwise conv
        filter_values = [
            [
                [
                    [127, -15, 109, 79, 91, -121, 44, 2, -20, 92, -17, -45, 115, -27, -127, -115],
                    [-9, 54, -73, -67, -2, -20, 15, -67, -3, -79, 68, -8, 91, 32, 13, -30],
                    [-7, -60, -92, -41, 84, 1, -96, 127, 19, -26, -73, 127, -104, 127, 30, 20]
                ],
                [
                    [-83, -12, 77, 10, 103, -127, 21, 66, -63, 58, -98, -34, 121, -35, -105, -127],
                    [10, -40, -127, -26, 36, 16, 7, 93, -1, -127, -113, 16, -28, -19, -13, -6],
                    [-11, 127, -98, 127, -21, -4, -117, 106, 19, -2, -127, -55, -127, 96, -5, -16]
                ],
                [
                    [62, -3, -25, 73, 127, -77, 25, 122, 127, 49, -97, -39, 83, -43, -36, -126],
                    [2, 42, -106, 93, 7, 12, 36, -1, 30, -117, -65, 22, 60, -16, -8, 4],
                    [13, -8, -28, -4, -92, 24, -127, -20, -12, 81, -125, 66, -115, 102, 5, -29]
                ]
            ]
        ]
        conv = tf.cast(conv, tf.float32)
        filters = tf.convert_to_tensor(filter_values, dtype=tf.float32)
        filters = tf.transpose(filters, perm=[1, 2, 3, 0])
        padding = "VALID"
        strides = [1, 1, 1, 1]
        depth_conv_tf = tf.nn.depthwise_conv2d(conv, filters, strides=strides, padding=padding)
        depth_conv_manual_2 = manual_depthwise_convolve_2(conv, filters, depth_conv_tf.shape, padding, strides)
        print(depth_conv_tf.shape)
        print(depth_conv_manual_2.shape)
        self.assertAllClose(depth_conv_tf, depth_conv_manual_2, rtol=1e-5, atol=1e-5)

# filter_array = np.array(filter_values, dtype=np.int8)


# Running the tests
if __name__ == "__main__":
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    tf.test.main()

# How to run the tests
# `python3 depthwise_conv.py``
# To launch a single test, run `python3 depthwise_conv.py TestDepthwiseConv.test_one_filter`
