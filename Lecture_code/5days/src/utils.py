import cv2

WINDOW_TITLE = "Show Image"


def show_img(img, title=WINDOW_TITLE, mul=1):
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW_TITLE, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_shape(img):
    h, w, c = img.shape
    print(f"height: {h}, width: {w}, channel: {c}")
