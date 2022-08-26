from math import ceil

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D, Dense,
                                     DepthwiseConv2D, Dropout,
                                     GlobalAveragePooling2D, Layer, Multiply,
                                     ReLU, Reshape)

BLOCKS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]


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


class MBConv(Layer):
    def __init__(
        self,
        filters_in,
        filters_out,
        kernel_size,
        strides=1,
        expand=1,
        is_skip=True,
        **kwargs
    ):
        self.filters = filters_in * expand
        self.filters_out = filters_out
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = "valid" if strides == 2 else "same"

    def call(self, inputs):
        # in dims = [batch, H, W, filters_in]
        # out dims = [batch, H/s, W/s, filters_out]
        x = inputs
        if 1 < self.expand:
            x = Conv2D(self.filters, kernel_size=(1, 1), strides=1, padding="same")(
                inputs
            )
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
        x = BatchNormalization(x)
        if self.is_skip:
            x = Add([x, inputs])
        return x


class MBCov1(Layer):
    def __init__(self, filters_in, filters_out, kernel_size, strides=1, **kwargs):
        self.filters = filters_in * 6
        self.filters_out = filters_out
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = "valid" if strides == 2 else "same"

    def call(self, inputs):
        # in dims = [batch, H, W, filters_in]
        # out dims = [batch, H/s, W/s, filters_out]
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
        self.output_dim

    def call(self, inputs):
        x = Conv2D(32, kernel_size=(3, 3), strides=2, padding="valid")(inputs)
        x = BatchNormalization()(x)
        x = Swish()(x)
        for block in BLOCKS:
            for _ in range(block["repeats"]):
                is_skip = True
                x = MBConv(
                    block["filters_in"],
                    block["filters_out"],
                    block["kernel_size"],
                    block["strides"],
                    block["expand_ratio"],
                    is_skip,
                )(x)
        x = Conv2D(1280, kernel_size=(1, 1), strides=1, padding="valid")(inputs)
        x = BatchNormalization()(x)
        x = Swish()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.output_dim, activation="softmax")(x)
