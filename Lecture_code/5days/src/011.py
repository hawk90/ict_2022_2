import cv2
import numpy as np
import utils

IMG_PATH = "../images"


if __name__ == "__main__":
    img = np.zeros((512, 512, 3), np.uint8)
    img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    utils.show_img(img)

    img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    utils.show_img(img)

    img = cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
    utils.show_img(img)

    img = cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
    utils.show_img(img)

# TODO:
# - logo.png 를 불러오기
# - 빨간 C 에 rectangle 노란색 220 X 200의 크기
# - 녹색 C 에 circle 파란색 내용물을 채우지 않고radius = 110
