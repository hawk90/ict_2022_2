#-*- coding:utf-8 -*-

import os  
import cv2 

import numpy as np    

IMG_PATH = "../images"


#TODO
# - logo.png 대신 다른 이지미 사용
# - 저장 

if __name__== "__main__":
    #img = cv2.imread(os.path.join(IMG_PATH, "logo.png"), cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(os.path.join(IMG_PATH, "coin.jpeg"), cv2.IMREAD_GRAYSCALE) 
    img = cv2.imread(os.path.join(IMG_PATH, "checkboard.png"), cv2.IMREAD_GRAYSCALE)  
    
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1 = 10, param2=5, minRadius=0, maxRadius=0) 
    #circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1 = 30, param2=25, minRadius=0, maxRadius=0)
    #circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100, param1 = 70, param2=25, minRadius=0, maxRadius=0) 
    
    circles = np.uint16(np.around(circles))
    
    for i in circles[0, :]:
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        
    cv2.imshow("img", cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #cv2.imwrite(os.path.join("../saveimages", "houghcircles_.png"), cimg)
    cv2.imwrite(os.path.join("../saveimages", "houghcircles_checkboard_2.png"), cimg) 
    
    
