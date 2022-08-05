import os 

import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

def equlization(img):
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        
        cdf = hist.cumsum()
        
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 /(cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")   
        return cdf[img]

IMG_PATH = "../images"

if __name__=="__main__":
    video = cv2.VideoCapture(os.path.join(IMG_PATH, "video.mp4"))
    
    while 1:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #equalization
        eqimg = equlization(gray)   
        img = np.vstack((gray, eqimg))      
        
        cv2.imshow("frame", img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
video.release()
cv2.destroyAllWindows()
    
    