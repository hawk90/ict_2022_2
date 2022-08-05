import cv2 
import numpy as np 

def equlization(img):
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        
        cdf = hist.cumsum()
        
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 /(cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")   
        return cdf[img]
        