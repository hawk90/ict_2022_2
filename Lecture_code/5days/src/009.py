# Pixel Normalization
import os

import cv2
import numpy as np

IMG_PATH = "../images"


if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "logo.png"))

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    # scale and shift by NORM_MINMAX
    dst = np.zeros(gray_img.shape, dtype=np.float32)
    cv2.normalize(gray_img, dst=dst, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    print(dst)
    cv2.imshow("NORM_MINMAX", np.uint8(dst * 255))

    # scale and shift by NORM_INF
    dst = np.zeros(gray_img.shape, dtype=np.float32)
    cv2.normalize(gray_img, dst=dst, alpha=1.0, beta=0, norm_type=cv2.NORM_INF)
    print(dst)
    cv2.imshow("NORM_INF", np.uint8(dst * 255))

    # scale and shift by NORM_L1
    dst = np.zeros(gray_img.shape, dtype=np.float32)
    cv2.normalize(gray_img, dst=dst, alpha=1.0, beta=0, norm_type=cv2.NORM_L1)
    print(dst)
    cv2.imshow("NORM_L1", np.uint8(dst * 10000000))

    # scale and shift by NORM_L2
    dst = np.zeros(gray_img.shape, dtype=np.float32)
    cv2.normalize(gray_img, dst=dst, alpha=1.0, beta=0, norm_type=cv2.NORM_L2)
    print(dst)
    cv2.imshow("NORM_L2", np.uint8(dst * 10000))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
