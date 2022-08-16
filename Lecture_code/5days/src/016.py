# Histogram Equalization
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "../images"


# TODO:
# - equalization 함수화
# - BGR Color Space 에서 Equalization
#   - Split B, G, R
#   - channel B에서 Equalization
#   - channel G에서 Equalization
#   - channel R에서 Equalization
#   - merge B, G, R
# - Histogram 을 출력하세요


if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "lena.jpeg"), cv2.IMREAD_GRAYSCALE)

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")

    img2 = cdf[img]
    plt.subplot(121), plt.imshow(img), plt.title("Original")
    plt.subplot(122), plt.imshow(img2), plt.title("Equalization")
    plt.show()
