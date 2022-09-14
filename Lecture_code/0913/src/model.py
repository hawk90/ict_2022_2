import tensorflow as tf
import pix2pix

INPUT_SHAPE = [128, 128, 3]
LAYER_NAME = [
    "block_1_expand_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "block_13_expand_relu",
    "block_16_project",
]

encoder = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False)

layers = [encoder.get_layer(name).output for name in LAYER_NAME]

down_stack = tf.keras.Model(inputs=encoder.input, outputs=layers)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]


def unet_model():
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    x = inputs

    skips = down_stack(x)
    x = skips[-1]  # block_16_project's out
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same")

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
