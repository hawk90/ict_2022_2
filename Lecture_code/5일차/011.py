#-*- coding:utf-8-*-
import os 
import cv2 
import numpy as np  
import utils 

IMG_PATH = "./images" 

if __name__ == "__main__":
    img = np.zeros((512, 512, 3), np.uint8)
    img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    utils.show_img(img)
    
    img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    utils.show_img(img)
    
    img = cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
    utils.show_img(img)
    
    img = cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
    utils.show_img(img)
  
 #TODO
 # - logo.png 를 불러오기
 # - 빨간 C 에 rectangle 노란색 220 * 200의 크기 
 # - 녹색 C 에 circle 파란색 내용물을 채우지 않고 radius = 110
    
    logo_img = cv2.imread(os.path.join(IMG_PATH, "logo.png")) 
    
    w, h, c = logo_img.shape 
    
    sx = 190
    sy = 25
        
    ex = sx + 215    
    ey = sy + 205
    
    img = cv2.rectangle(logo_img, (sx, sy), (ex, ey), (0, 255, 255), 2)
    
    cx = 180
    cy = 340   
    img = cv2.circle(logo_img, (cx, cy), 105, (255, 0, 0), 1) 
    utils.show_img(img) 