# Pixel Logical Operation
import cv2
import numpy as np
import utils

OP_FUNCS = [cv2.bitwise_or, cv2.bitwise_and]


def call_ops(img1, img2, ops):
    if not ops:
        return

    for op in ops:
        output_img = op(img1, img2)
        utils.show_img(output_img)


if __name__ == "__main__":
    img1 = np.zeros(shape=[400, 400, 3], dtype=np.uint8)
    img1[100:200, 100:200, 1] = 255
    img1[100:200, 100:200, 2] = 255
    # utils.show_img(img1)

    img2 = np.zeros(shape=[400, 400, 3], dtype=np.uint8)
    img2[150:250, 150:250, 2] = 255
    # utils.show_img(img2)

    call_ops(img1, img2, OP_FUNCS)
