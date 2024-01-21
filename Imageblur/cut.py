import cv2
import os
import numpy as np

img_path = "image/"
out_img_path = "blur/"
cp_img_path = "cp_image/"
ep_img_path = "ep_image/"
path_list = [img_path]

for path in path_list:
    image_list = os.listdir(path)
    for image in image_list:
        img = cv2.imread(path+image)
        if len(image)==9:
            os.remove(path+image)
        elif img.shape[0] < 740 or img.shape[1] <650 :
            img = cv2.resize(img,dsize=(650,740))
            print(image)
        else:
            img = img[0:740,0:650]
        cv2.imwrite(path+image,img)

txt = []
for i in range(2490):
    txt.append((img_path+str(i+1).zfill(4),out_img_path+str(i+1).zfill(4),cp_img_path+str(i+1).zfill(4),ep_img_path+str(i+1).zfill(4)))
np.savetxt('test.txt',txt,fmt='%s')