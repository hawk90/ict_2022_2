   
def call_ops(img1, img2, ops):
    if not ops:
        return 
    
    for op in ops:
        output_img = op(img1, img2)
        utils.show_img(output_img)
        
if __name__ == "__main__":
    img1 = np.zeros(shape=[400, 400, 3], dtype=np.uint8
                    
)

    img1[100:200, 100:200, 1] = 255
    img1[100:200, 100:200, 2] = 255
    utils.show_img(img1)
    print("a3")
    
    img2 = np.zeros(shape=[400, 400, 3], dtype=np.uint8)
    print("a3")
    utils.show_img(img2)
    print("a5")
    call_ops(img1, img2, OP_FUNCS)    
    print("a6") 