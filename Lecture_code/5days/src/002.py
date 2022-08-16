# Color 2 Gray
import os

import cv2
import utils

IMG_PATH = "../images"


if __name__ == "__main__":
    color_img = cv2.imread(os.path.join(IMG_PATH, "summit.jpg"))
    utils.show_img(color_img, "Color Image")

    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    utils.show_img(gray_img, "Convert Gray Image")

    #
    load_img = cv2.imread(os.path.join(IMG_PATH, "summit.jpg"), cv2.IMREAD_GRAYSCALE)
    utils.show_img(gray_img, "Load Gray Image")
