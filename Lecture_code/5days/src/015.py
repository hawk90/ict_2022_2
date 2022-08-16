# Image Pyramids
import os

import cv2

IMG_PATH = "../images"

# TODO:
# - 새로운 img 불러와서
# - lower를 한번 더 lower 하세요
# - higher를 한번 더 higher 하세요
# - imwrite로 모두 저장 하세요

if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "lena.jpeg"))

    lower_reso = cv2.pyrDown(img)
    higher_reso = cv2.pyrUp(img)

    cv2.imshow('img', img)
    cv2.imshow('lower', lower_reso)
    cv2.imshow('higher', higher_reso)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
