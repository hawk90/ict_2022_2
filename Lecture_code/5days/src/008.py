# Pixel Speration
import os

import cv2
import numpy as np
import utils

IMG_PATH = "../images"


def min_max_loc(img):
    min, max, min_loc, max_loc = cv2.minMaxLoc(img)
    print(f"Min: {min}, Max: {max}")
    print(f"Min Loc: {min_loc}, Max: {max_loc}")
    return min, max, min_loc, max_loc


def mean_stddev(img):
    mean, stddev = cv2.meanStdDev(img)
    print(f"Mean: {mean}, Stddev: {stddev}") # TODO: f-string format, 소수점 2번쨰 자리까지 표기
    return (mean, stddev)


def binary_img(img, threshold=100):
    threshold, _ = mean_stddev(img)
    img[np.where(img < threshold)] = 0
    img[np.where(img >= threshold)] = 255
    print(img)
    return img


if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "summit.jpg"), cv2.IMREAD_GRAYSCALE)
    utils.show_img(img)

    min_max_loc(img)

    bin_img = binary_img(img)
    utils.show_img(img)
