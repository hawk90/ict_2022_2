from math import ceil

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     DepthwiseConv2D, Dropout,
                                     GlobalAveragePooling2D, Layer, Multiply,
                                     ReLU, Reshape)


class Swish(Layer):
    def __init__(self, **kwargs):
        pass

    def call(self, x):
        return x * ReLU()(x + 3) / 6


class SE(Layer):
    def __init__(self, filters, reduction_ratio=4, **kwargs):
        self.filters = filters
        self.reduction_ratio = reduction_ratio
        self.se_filters = max(1, int(filters / reduction_ratio))

    def call(self, inputs):
        # in dims = [batch, H, W, channels]
        # out dims = [batch, 1, 1, channels]        @GlobalAveragePooling2D
        # out dims = [batch, 1, 1, channels/ratio]  @Conv2D_1
        # out dims = [batch, 1, 1, channels]        @COnv2D_2
        x = GlobalAveragePooling2D(inputs, keepdims=True)(inputs)
        x = Conv2D(self.se_filters, strides=1, activation="relu", padding="same")(x)
        x = Conv2D(self.filters, strides=1, activation="sigmoid", paddig="same")(x)
        return Multiply()([inputs, x])


class MBCov1(Layer):
    def __init__(self, **kwargs):
        pass

    def call(self, inputs):
        x = DepthwiseConv2D()(input)
        x = BatchNormalization()(x)
        x = Swish()(x)
        x = SE()(x)
        x = Conv2D()(x)
        return BatchNormalization()(x)


class MBCov6(Layer):
    def __init__(self, filters_in, filters_out, kernel_size, strides=1, **kwargs):
        self.filters = filters_in * 6
        self.filters_out = filters_out
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = "valid" if strides == 2 else "same"

    def call(self, inputs):
        # NOTE: Layer에 필요한 args(filters, kernel_size, strides, padding)
        # NOTE: filters == channel of out_dims
        # NOTE: ((W - k + 2P) / S) + 1
        # NOTE: W: width, k: kernel size, P: padding, S: stride

        # in dims = [batch, H, W, filters_in]
        # out dims = [batch, H/s, W/s, filters_out]
        x = Conv2D(self.filters, kernel_size=(1, 1), strides=1, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Swish()(x)
        x = DepthwiseConv2D(
            self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )(x)
        x = BatchNormalization()(x)
        x = Swish()(x)
        x = SE(self.filters)(x)
        x = Conv2D(self.filters_out, kernel_size=(1, 1), strides=1, padding="same")(x)
        return BatchNormalization()(x)


class Efficient(Model):
    def __init__(self, output_dim, **kwargs):
        pass

    def call(self, inputs):
        pass
