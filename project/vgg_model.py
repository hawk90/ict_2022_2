from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

class VGG(Model):
    def __init__(self, **kwargs):
        super(VGG, self).__init__(**kwargs)
        self.input_conv = Conv2D(
            input_shape=(32,32,3),
            filters=64,
            kernel_size=(3,3),
            padding="same",
            activation="relu"
        )

        self.conv_1 = Conv2D(
            filters=64, kernel_size=(3,3), padding="same", activation="relu"        
        )

        self.max_pool = MaxPool2D(pool_size=(2,2), strides=(2,2))

        self.conv_2 = Conv2D(
            filters=128, kernel_size=(3,3), padding="same", activation="relu"        
        )

        self.conv_3 = Conv2D(
            filters=256, kernel_size=(3,3), padding="same", activation="relu"        
        )

        self.conv_4 = Conv2D(
            filters=512, kernel_size=(3,3), padding="same", activation="relu"        
        )

        self.flatten = Flatten()

        self.dense = Dense(units=4096, activation="relu")
        self.out_dense = Dense(units=10)

    def call(self, inputs):
        x = self.input_conv(inputs)
        x = self.conv_1(x)
        x = self.max_pool(x)

        x = self.conv_2(x)
        x = self.conv_2(x)
        x = self.max_pool(x)

        x = self.conv_3(x)
        x = self.conv_3(x)
        x = self.conv_3(x)
        x = self.max_pool(x)

        x = self.conv_4(x)
        x = self.conv_4(x)
        x = self.conv_4(x)
        x = self.max_pool(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.dense(x)
        x = self.out_dense(x)

        return x