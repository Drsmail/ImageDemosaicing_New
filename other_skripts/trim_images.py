import numpy
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import cv2


def mozaik(image_path):
    img_BGR = cv2.imread(image_path)
    B_x, G_x, R_x = cv2.split(img_BGR)


    # Стираем информацию о цвете в нужных местах.
    G_x[::2, ::2] = 0
    G_x[1::2, 1::2] = 0

    B_x[::2, 1::2] = 0
    B_x[1::2, :] = 0

    R_x[::2, :] = 0
    R_x[1::2, 1::2] = 0

    # Собираем RGB with (H, W, C)
    x = cv2.merge((B_x, G_x, R_x))
    cv2.imwrite(image_path,x)



FileList = []
path_to_trim_data = 'F:/Unpaked_dataset/images/test/trim'

for dir_name, _, file_list in os.walk(path_to_trim_data):
    for file in file_list:
        if 'png' in file:
            FileList.append(os.path.join(dir_name, file))

for file in FileList:
    mozaik(file)