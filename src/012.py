import os
import cv2
import numpy as np

IMG_PATH = "../images"

def nothing(x):
    pass

if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "lenna.jpg"))

    for k in range(20):
        if k == 0:
            continue
        cv2.namedWindow('image k: {k}')

        kernel = np.ones((k,k), np.float32) / (k * 2)
        dst = cv2.filter2D(img -1, kernel)

        cv2.imshow(f'K: {k}', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()