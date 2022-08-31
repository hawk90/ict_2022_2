import logging

import cv2

WINDOW_TITLE = "Show Image"


def show_img(img, title=WINDOW_TITLE, mul=1):
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW_TITLE, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class BBox:
    def __init__(self, color=(0, 255, 0), thickness=1):
        self.color = color
        self.thickness = thickness

    def show(self, img, coordinates):
        for coord in coordinates:
            start, end = coord
            img = cv2.rectangle(img, start, end, self.color, self.thickness)
        show_img(img)


class Logger:
    def __init__(self, level=logging.ERROR):
        # NOTE: CRITICAL    (level high)
        # NOTE: ERROR
        # NOTE: WARNING
        # NOTE: INFO
        # NOTE: DEBUG       (level low)

        self.logger = logging.getLogger()

        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(levelname)s] - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger
