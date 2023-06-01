import numpy as np
import cv2
import matplotlib.pyplot as plt
from Mozaik import *

from quality_control import PSNR, SSIM

# reads image
# img_BGR = cv2.imread('Images/2colors.jpg')
img_BGR = cv2.imread('../Images/pepe.jpg')
#img_BGR = cv2.imread('Images/pixel_64_64.jpg')

img_RGB = img_BGR[:, :, ::-1] #reverse BGR to RGB

# extract channels
b, g, r = cv2.split(img_BGR)
zer = np.zeros(b.shape, dtype=b.dtype)

r_m, g_m, b_m = mozaik(r, g, b)

plt.figure(num=2)

val = 0.25
green_kurnel = np.array([[0, val, 0],
                         [val,1,val],
                         [0,val,0]])

red_kurnel = np.array([  [val, 0.5, val],
                         [0.5, 1, 0.5],
                         [val, 0.5, val]])

blue_kurnel = np.array([  [val, 0.5, val],
                         [0.5, 1, 0.5],
                         [val, 0.5, val]])

g_r = convolution2d(g_m, green_kurnel, 0).astype(np.uint8)
r_r = convolution2d(r_m, red_kurnel, 0).astype(np.uint8)
b_r = convolution2d(b_m, blue_kurnel, 0).astype(np.uint8)


plt.subplot(121)
plt.imshow(cv2.merge((r_m, zer, zer)))
plt.subplot(122)
plt.imshow(cv2.merge((r_r, zer , zer)))




plt.subplot(121)
plt.imshow(cv2.merge((r, g, b)))
plt.title("Green Channel")
plt.subplot(122)
plt.imshow(cv2.merge((r_m, g_m, b_m)))
plt.title("Convolve Channel")

plt.show()


psnr_g = PSNR(g, g_r)
print(f"PSNR value is {psnr_g } dB")

ssim_g =SSIM(g, g_r,
                  data_range=g_r.max() - g_r.min())
print(f"SSIM value is {ssim_g}")



plt.figure(num=1, figsize=[20, 5])
# Каналы оригинального изоброжения
plt.subplot(341)
plt.imshow(cv2.merge((r, zer, zer)))
plt.title("Red Channel")
plt.subplot(342)
plt.imshow(cv2.merge((zer, g, zer)))
plt.title("Green Channel")
plt.subplot(343)
plt.imshow(cv2.merge((zer, zer, b)))
plt.title("Blue Channel")
# Собираем по 3м каналам
plt.subplot(344)
imgMerged = cv2.merge((r, g, b))
plt.imshow(imgMerged)
plt.title("Merged Output")

# RAW каналы
plt.subplot(345)
plt.imshow(cv2.merge((r_m, zer, zer)), )
plt.subplot(346)
plt.imshow(cv2.merge((zer, g_m, zer)))
plt.subplot(347)
plt.imshow(cv2.merge((zer, zer, b_m)))
# Собираем по 3м каналам
plt.subplot(348)
imgMerged_m = cv2.merge((r_m, g_m, b_m))
plt.imshow(imgMerged_m)


# Востановленные каналы
plt.subplot(349)
plt.imshow(cv2.merge((r_r, zer, zer)), )
plt.subplot(3,4,10)
plt.imshow(cv2.merge((zer, g_r, zer)))
plt.subplot(3,4,11)
plt.imshow(cv2.merge((zer, zer, b_r)))
# Собираем по 3м каналам
plt.subplot(3,4,12)
imgMerged_r = cv2.merge((r_r, g_r, b_r))
plt.imshow(imgMerged_r)


plt.show()
