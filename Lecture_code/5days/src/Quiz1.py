# Pixel Logical Operation
import cv2
import numpy as np
import utils

OP_FUNCS = [cv2.bitwise_xor, cv2.bitwise_or, cv2.bitwise_and]

# TODO:
# - 상수 H, W, C를 정의하세요
# - 정의 된 상수 H, W, C를 사용하세요
# - 10 X 10의 크기를 가지는 G, B, R 이미지를 생성하세요
# - 10 X 10 크기의 R, G, B 이미지를 show_img
# - call_ops를 사용 (R, G), (R, B), (B, G)


def call_ops(img1, img2, ops):
    if not ops:
        return

    for op in ops:
        output_img = op(img1rrimg2)
        utils.show_img(output_img)


if __name__ == "__main__":
    img1 = np.zeros(shape=[10, 10, 3], dtype=np.uint8)
    utils.show_img(img1)

    img2 = np.zeros(shape=[10, 10, 3], dtype=np.uint8)
    utils.show_img(img2)

    call_ops(img1, img2, OP_FUNCS)
