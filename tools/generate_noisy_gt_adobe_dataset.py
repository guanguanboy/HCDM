import numpy as np
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import skimage
import random

import utils_image as util
import random

import scipy
import scipy.stats as ss
import scipy.io as io
from scipy import ndimage
from scipy.interpolate import interp2d

import numpy as np
import torch
from utils_sisr import add_blur,add_blur_isotropic,add_blur_to_img,add_blur_to_img_isotropic

def addGaussNoise(s):
    #var = random.uniform(0.0001, 1.0)
    var = 0.25
    noisy = skimage.util.random_noise(s, mode='gaussian', var=var)
    return noisy



sub_dataset_list = ['/data1/liguanlin/Datasets/iHarmony/HAdobe5k_1024']

for sub_dataset_name in sorted(sub_dataset_list):
    
    gt_file_path = sub_dataset_name + '/real_images_test_samll_full/' #原始合成图像路径
    gt_file_list = os.listdir(gt_file_path) #获取合成图像的list

    for image in sorted(gt_file_list):
        img_path = gt_file_path + image #原始合成图像地址
        print(img_path)
        gt = Image.open(img_path).convert('RGB') #读取合成图像
        numpy_gt = np.array(gt) #将合成图像转换为numpy格式

        noisy_gt = addGaussNoise(numpy_gt.view()) #添加噪声，生成全图噪声
        numpy_noisy_gt = np.array(noisy_gt*255, dtype='uint8') #将噪声图转换numpy格式
        #生成一张空图，用来接受原始合成图的背景和噪声全图的前景

        #使用Bilinear的方式下采样为256*256
        #生成带噪图的存储文件夹
        save_blur_img_path = sub_dataset_name + '/real_images_test_noised_25_small_256_gt/'
        if not os.path.exists(save_blur_img_path):
            os.mkdir(save_blur_img_path)
        blur_img_name = save_blur_img_path + image #带噪图的image地址

        save_noised = Image.fromarray(numpy_noisy_gt)
        #使用Bilinear的方式下采样为256*256
        new_size = (256, 256) # 缩小到原来的一半
        img_reisized = save_noised.resize(new_size, resample=Image.BILINEAR)
        img_reisized.save(blur_img_name)

