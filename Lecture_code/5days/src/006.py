# Pixel Speration
import os

import cv2
import numpy as np
import utils

IMG_PATH = "../images"


if __name__ == "__main__":
    input_img = cv2.imread(os.path.join(IMG_PATH, "logo.png"))
    utils.show_img(input_img)

    (b, g, r) = cv2.split(input_img)
    for i, channel in enumerate((b, g, r)):
        channel_img = np.zeros(shape=input_img.shape, dtype=input_img.dtype)
        channel_img[:, :, i] = channel
        utils.show_img(channel_img)

    blue = np.zeros(shape=[300, 300], dtype=np.uint8)
    green = np.zeros(shape=[300, 300], dtype=np.uint8)
    red = np.zeros(shape=[300, 300], dtype=np.uint8)

    blue[:, :] = 204
    green[:, :] = 204
    red[:, :] = 255
    created_img = cv2.merge((blue, green, red))
    utils.show_img(created_img)
