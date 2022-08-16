# Color Space
import os

import cv2
import utils

IMG_PATH = "../images"


if __name__ == "__main__":
    input_img = cv2.imread(os.path.join(IMG_PATH, "arithmetic2.jpg"))
    utils.show_img(input_img)

    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    utils.show_img(hsv, "HSV")

    yuv = cv2.cvtColor(input_img, cv2.COLOR_BGR2YUV)
    utils.show_img(yuv, "YUV")

    ycrcb = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)
    utils.show_img(ycrcb, "YCrCb")

###################################################
# TODO:
# - 상수SAVE_PATH = "../save"
# - os.mkdir() 함수를 이용하여 ../save 폴더 만들기
# - cv2.imwrite() 함수를 이용하여 SAVE_PATH를 사용하여 ../save 폴더 아래 이미지를 저장
# - os.path.join 을 사용
#
# Syntax: cv2.imwrite(filename, image)
#
# example) cv2.imwrite("../save/XXX.jpg", input_img)
