import tensorflow as tf
from tensorflow.kears.activats import relu
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     DepthwiseConv2D, Dropout,
                                     GlobalAveragePooling2D, Layer, Reshape)


class Swish(Layer):
    def __init__(self, **kwargs):
        pass

    def call(self, x):
        return x * relu(x + 3) / 6


class SE(Layer):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        pass

    def call(self, inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = Reshape(1, 1, self.filters)(x)
        x = Conv2D((1, 1, self.filters), strides=1, activation="relu", padding="same")(
            x
        )
        x = Conv2D(self.filters, strides=1, activation="sigmoid", paddig="same")(x)
        pass


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
    def __init__(self, **kwargs):
        pass

    def call(self, inputs):
        x = Conv2D()(inputs)
        x = Swish()(x)
        x = BatchNormalization()(x)
        x = DepthwiseConv2D()(x)
        x = BatchNormalization()(x)
        x = Swish()(x)
        x = SE()(x)
        x = Conv2D()(x)
        return BatchNormalization()(x)


class Efficient(Model):
    def __init__(self, output_dim, **kwargs):
        pass

    def call(self, inputs):
        pass
