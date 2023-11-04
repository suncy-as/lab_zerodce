import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
import tifffile as tiff
random.seed(1143)


def populate_train_list(lowlight_images_path):

    image_list_lowlight = glob.glob(lowlight_images_path + "*.tif")

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


def get_img(path):  # 读取tiff格式和其它格式图片，主要为3维16位图片
    if 'tif' in path:
        img = tiff.imread(path)
    else:
        img = 0
        print('文件格式好像有问题不是tiff')
    img = torch.tensor(img / 65536)

    img = img[np.newaxis, :, :]  # 为了减少运行内存只能这样写了
    print('img.shape:{}'.format(img.shape))
    return img

class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path):

        self.train_list = populate_train_list(lowlight_images_path) 
        self.size = 256

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        #         data_lowlight = Image.open(data_lowlight_path)    这一句要改
        data_lowlight = get_img(data_lowlight_path)

        return data_lowlight


    def __len__(self):
        return len(self.data_list)

