import cv2
from utils import BBox

IMG_PATH = "../data/VOC2012/JPEGImages/2007_000027.jpg"

if __name__ == "__main__":
    bbox = BBox()

    img = cv2.imread(IMG_PATH)
    coordinate = ((174, 101), (349, 351))
    bbox.show(img, coordinate)
