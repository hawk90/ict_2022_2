import os  

import cv2 
import numpy as np  
import utils 

IMG_PATH = "./images"

if __name__ == "__main__":
    input_img = cv2.imread(os.path.join(IMG_PATH, "logo.png"))
    utils.show_img(input_img)
    
    (b, g, r) = cv2.split(input_img)
    for i, channel in enumerate((b, g, r)):
        channel_img = np.zeros(shape=input_img.shape, dtype=input_img.dtype)
        print(channel)
        channel_img[:, :, i] = channel 
        utils.show_img(channel_img) 
        
        