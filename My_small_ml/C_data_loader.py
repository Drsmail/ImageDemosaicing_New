import numpy
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import cv2

    #pinmemory, persistanceworkers

class MozaikDataset(Dataset):

    def mozaik(self, image_path):

        img_BGR = cv2.imread(image_path)
        B_y, G_y, R_y = cv2.split(img_BGR)

        # Создаём копии, так как аргументы это ссылки
        R_x = R_y.copy()
        G_x = G_y.copy()
        B_x = B_y.copy()

        # Стираем информацию о цвете в нужных местах.
        G_x[::2, ::2] = 0
        G_x[1::2, 1::2] = 0

        B_x[::2, 1::2] = 0
        B_x[1::2, :] = 0

        R_x[::2, :] = 0
        R_x[1::2, 1::2] = 0

        # Собираем RGB with (H, W, C)
        x = cv2.merge((R_x, G_x, B_x))
        y = cv2.merge((R_y, G_y, B_y))

        # conver to tensor order,  tensor Image is a tensor with (C, H, W)
        # x = np.moveaxis(x, -1, 0)
        # y = np.moveaxis(y, -1, 0)

        return x,y

    def trim_fun(self, index):

        img_path_x = self.FileList[index]

        img_BGR_x = cv2.imread(img_path_x)
        B_x, G_x, R_x = cv2.split(img_BGR_x)
        x_label = cv2.merge((R_x, G_x, B_x))

        img_path_y = self.TrimFileList[index]

        img_BGR_y = cv2.imread(img_path_y)
        B_y, G_y, R_y = cv2.split(img_BGR_y)
        y_label = cv2.merge((R_y, G_y, B_y))



        if self.transform:
            x_label = self.transform(x_label)
            y_label = self.transform(y_label)

        return (x_label, y_label)


    def __init__(self, path_to_data, transform = None, split = 1, trim=False):
        # Initialize data, download, etc.
        # read with numpy or pandas
        self.path_to_data = path_to_data
        self.transform = transform
        self.FileList = []
        self.TrimFileList = []


        for dir_name, _, file_list in os.walk(self.path_to_data):
            for file in file_list:
                if 'png' in file:
                    self.FileList.append(os.path.join(dir_name, file))

        if trim:

            self.TrimFileList = self.TrimFileList[::split]

            self.__getitem__ = self.trim_fun

            for dir_name, _, file_list in os.walk(self.path_to_data + '/trim'):
                for file in file_list:
                    if 'png' in file:
                        self.TrimFileList.append(os.path.join(dir_name, file))

        #print (f"Len before split: {len(self.FileList)}")
        if split != 1:
            self.FileList = self.FileList[::split]

        #print(f"Len after split: {len(self.FileList)}")

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):

        img_path = self.FileList[index]
        x_label, y_label = self.mozaik(img_path)

        # x_label = np.random.randint(250, size=(128, 128, 3), dtype='uint8')
        # y_label = np.random.randint(250, size=(128, 128, 3), dtype='uint8')

        if self.transform:
            x_label = self.transform(x_label)
            y_label = self.transform(y_label)

        return (x_label, y_label)

    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.FileList)

# a = MozaikDataset('F:/Unpaked_dataset/images/test')


