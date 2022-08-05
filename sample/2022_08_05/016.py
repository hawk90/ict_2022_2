import os 

import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

IMG_PATH = "../images"

#TODO

def equlization(img):
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        
        cdf = hist.cumsum()
        
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 /(cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")   
        return cdf[img]
        
        

if __name__=="__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "lena.png"), cv2.IMREAD_GRAYSCALE)
    
    img2 = equlization(img)
    
    plt.subplot(121), plt.imshow(img), plt.title("Original")
    plt.subplot(122), plt.imshow(img2), plt.title("Equalization")
    plt.show()
    
    

  