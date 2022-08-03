
#-*- coding:utf-8-*-
import cv2
import numpy as np 
import utils 

OP_FUNCS = [cv2.bitwise_xor, cv2.bitwise_or, cv2.bitwise_and]
#   TODO
# - 상수 H, W, C를 정의 하세요
# - 정의된 상수 H, W, C를 사용하세요
# - 10 * 10의 크기를 가지는 G, B, R이미지를 생성하세요
# - 10*10 크기의 R, G, G 이미지를 show_img
# - call_op를 사용 (R, G), (R, B), (G, B)

H = 30
W = 30
C = 3

OP_FUNCS = [cv2.bitwise_xor, cv2.bitwise_or, cv2.bitwise_and]
   
def call_ops(img1, img2, ops):
    if not ops:
        return 
    
    for op in ops:
        output_img = op(img1, img2)
        utils.show_img(output_img)
        
if __name__ == "__main__":
    imgr = np.zeros(shape=[W, H, C], dtype=np.uint8 )
    imgg = np.zeros(shape=[W, H, C], dtype=np.uint8 )
    imgb = np.zeros(shape=[W, H, C], dtype=np.uint8 ) 


    imgr[10:20, 10:20, 0] = 255
    imgg[10:20, 10:20, 1] = 255
    imgb[10:20, 10:20, 2] = 255   

    utils.show_img(imgr)
    utils.show_img(imgg)
    utils.show_img(imgb)
    
    call_ops(imgr, imgg, OP_FUNCS)   
    call_ops(imgr, imgb, OP_FUNCS)    
    call_ops(imgg, imgb, OP_FUNCS)     