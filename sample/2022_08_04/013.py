
#-*- coding:utf-8-*-
import os 

import cv2
import numpy as np 

from matplotlib import pyplot as plt 

IMG_PATH = "../images" 

if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "lena.png"))
    
    # pyplot 를 사용하기 위해서 BGR을 RGB로 변환.
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    
    dst1 = cv2.blur(img, (7, 7))
    
    dst2 = cv2.GaussianBlur(img, (5, 5), 0)
    
    dst3 = cv2.medianBlur(img, 9)
    
    dst4 = cv2.bilateralFilter(img, 9, 75, 75)
    
    images = [img, dst1, dst2, dst3, dst4]
    
    titles = ["Original", "Blur(7*7)", "Gaussian Blur(5*5)", "Median Blur", "Bilateral"]
    
    for i in range(5):
        plt.subplot(3, 2,  i+ 1), plt.imshow(images[i]), plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    #TODO 
    i = 0
    for img, title in zip():
        plt.subplot(3, 3,  i+1)
        plt.imshow(img)
        plt.title(title)
        i = i+1
        plt.xtricks([])
        plt.ytcks([])    
    plt.show()
        
        