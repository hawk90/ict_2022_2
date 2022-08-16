# Pixel arithmetic
import os

import cv2
import numpy as np
import utils

IMG_PATH = "../images"


def img_add(img1, img2):
    pass


def img_sub(img1, img2):
    pass


def img_multiply(img1, img2):
    pass


def img_divide(img1, img2):
    pass


if __name__ == "__main__":
    img1 = cv2.imread(os.path.join(IMG_PATH, "arithmetic1.jpg"))
    img2 = cv2.imread(os.path.join(IMG_PATH, "arithmetic2.jpg"))

    utils.print_shape(img1)
    utils.print_shape(img2)

    buff = np.zeros(img1.shape, img1.dtype)
    utils.print_shape(buff)

    # add
    buff = np.zeros(img1.shape, img1.dtype)
    cv2.add(img1, img2, buff)
    utils.show_img(buff)

    # sub
    buff = np.zeros(img1.shape, img1.dtype)
    cv2.subtract(img1, img2, buff)
    utils.show_img(buff)

    # mul
    buff = np.zeros(img1.shape, img1.dtype)
    cv2.multiply(img1, img2, buff)
    utils.show_img(buff)

    # div
    buff = np.zeros(img1.shape, img1.dtype)
    cv2.divide(img1, img2, buff)
    utils.show_img(buff)
