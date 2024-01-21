from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import os

img_path = "results/MIMO-UNetPlus/result_image/"
data_path = "training_set/test/sharp/"
img_list = os.listdir(img_path)

s = 0.0
p = 0.0
c = 0

if __name__ == "__main__":
	# If the input is a multichannel (color) image, set multichannel=True.
    for img in img_list:
        #print(img)
        img1 = np.array(Image.open(data_path+img))
        img2 = np.array(Image.open(img_path+img))
        s += ssim(img1, img2, channel_axis=2)
        p += psnr(img1, img2)
        c+=1
    print(s/c,p/c)
