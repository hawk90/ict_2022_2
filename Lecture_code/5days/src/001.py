# Load Image
import os

import cv2
import utils

IMG_PATH = "../images"


if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "summit.jpg"))

    utils.show_img(img)
