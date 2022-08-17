import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, GlobalAveragePooling2D,
                                     Layer, MaxPool2D)


class Stem(Layer):
    def __init__(self, **kwargs):
        super(Stem, self).__init__(**kwargs)
        self.hidden = [
            Conv2D(
                filters=64,
                kernel_size=(7, 7),
                strides=(2, 2),
                activation="relu",
                padding="same",
            ),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            BatchNormalization(),
            Conv2D(filters=64, kernel_size=(1, 1), activation="relu", padding="valid"),
            Conv2D(filters=192, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return x


class InceptionModule(Layer):
    def __init__(self, inception_filters, **kwargs):
        super(InceptionModule, self).__init__(**kwargs)
        # INFO: [1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, 1x1_pool]
        (
            filters_1x1,
            filters_3x3_reduce,
            filters_3x3,
            filters_5x5_reduce,
            filters_5x5,
            filters_1x1_pool,
        ) = inception_filters

        self.paths = [
            [
                Conv2D(
                    filters=filters_1x1,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                )
            ],
            [
                Conv2D(
                    filters=filters_3x3_reduce,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                ),
                Conv2D(
                    filters=filters_3x3,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                ),
            ],
            [
                Conv2D(
                    filters=filters_5x5_reduce,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                ),
                Conv2D(
                    filters=filters_5x5,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding="same",
                ),
            ],
            [
                MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same"),
                Conv2D(
                    filters=filters_1x1_pool,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                ),
            ],
        ]
        self.depth_concat = tf.concat

    def call(self, inputs):
        x = inputs
        path_outs = []
        for path in self.paths:
            for layer in path:
                x = layer(x)
            path_outs.append(x)
        return self.depth_concat(path_outs, axis=3)


class InceptionV1(Model):
    def __init__(self, output_dim, **kwargs):
        super(InceptionV1, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden = [
            Stem(name="stem"),
            # INFO: [1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, 1x1_pool]
            InceptionModule([64, 96, 128, 16, 32, 32], name="inception_3a"),
            InceptionModule([128, 128, 192, 32, 96, 64], name="inception_3b"),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            InceptionModule([192, 96, 208, 16, 48, 64], name="inception_4a"),
            InceptionModule([160, 112, 224, 24, 64, 64], name="inception_4b"),
            InceptionModule([128, 128, 256, 24, 64, 64], name="inception_4c"),
            InceptionModule([112, 144, 288, 32, 64, 64], name="inception_4d"),
            InceptionModule([256, 160, 320, 32, 128, 128], name="inception_4e"),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            InceptionModule([256, 160, 320, 32, 128, 128], name="inception_5a"),
            InceptionModule([384, 192, 384, 48, 128, 128], name="inception_5b"),
            GlobalAveragePooling2D(),
            Dropout(rate=0.4),
        ]
        self.out = Dense(10)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)


# https://www.tensorflow.org/guide/keras/custom_layers_and_models
