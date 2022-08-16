import os

import cv2

IMG_PATH = "../images"

# TODO:
# - equlization을 하여 원 영상과 비교하세요
# - np.vstack

if __name__ == "__main__":
    video = cv2.VideoCapture(os.path.join(IMG_PATH, "video.mp4"))

    while 1:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        cv2.imshow("frame", gray)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()
