# INFO: Generator
# INFO:  - latent vector [200]
# INFO:  - 4 x 4 x 1024
# INFO:  - 8 x 8 x 512
# INFO:  - 16 x 16 x 256
# INFO:  - 32 x 32 x 128
# INFO:  - 64 x 64 x 64
# INFO:  - 128 x 128 x 3

# INFO: Discriminator
# INFO:  - 128 x 128 x 3
# INFO:  - 64 x 64 x 64
# INFO:  - 32 x 32 x 128
# INFO:  - 16 x 16 x 256
# INFO:  - 8 x 8 x 512
# INFO:  - 4 x 4 x 1024     kernel=2, stride=2
# INFO:  - 1                (sigmoid)

# INFO: ReLU, BatchNormalization


import tensorflow as tf
from tensorflow.keras import layers


class _UpLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(self).__init__()


class Generator(tf.keras.Model):
    def __init__(self):
        super(self).__init__()


class _DownLayer(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(self).__init__()
        self.filters = filters

    def build(self, input_shape):
        self.conv2d = layers.Conv2D(
            self.filters, kernel_size=3, stride=2, padding="same"
        )
        self.bn = layers.BatchNormalization()
        self.activation = layers.ReLU()
        super(self).build(input_shape)

    def call(self, inputs):
        x = inputs
        x = self.conv2d(x)
        x = self.bn(x)
        return self.activation(x)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(self).__init__()
        pass
