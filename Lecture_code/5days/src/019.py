# Segmentation
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "../images"


if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "coin.jpeg"))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img[markers == -1] = [255, 0, 0]

    images = [gray, thresh, sure_bg, dist_transform, sure_fg, unknown, markers, img]
    titles = [
        "Gray",
        "Binary",
        "Sure BG",
        "Distance",
        "Sure FG",
        "Unknow",
        "Markers",
        "Result",
    ]

    for i in range(len(images)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()
