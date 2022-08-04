import os 

import cv2
from matplotlib import pyplot as plt 

IMG_PATH = "../images" 

def image_hist(image):
    cv2.imshow("input", image)
    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
        
    plt.show()
    
if __name__ == "__main__":
    #src = cv2.imread(os.path.join(IMG_PATH, "logo.png"))
    src = cv2.imread(os.path.join(IMG_PATH, "image0.png")) 
    
    cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.imshow("input", gray)
    
    #custom_hist(gray)
    image_hist(src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    