# Hough Circle
import os

import cv2
import numpy as np

IMG_PATH = "../images"

# TODO:
# - logo.png 대신 다른 이미지 사용
# - 저장

if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "logo.png"), cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=25, minRadius=0, maxRadius=0
    )

    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("img", cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
