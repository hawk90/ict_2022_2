from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D


class VGG(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(units=4096, activation="relu"),
            Dense(units=4096, activation="relu"),
        ]
        self.out = Dense(units=10)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)
