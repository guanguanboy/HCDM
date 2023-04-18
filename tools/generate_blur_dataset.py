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
from utils_sisr import add_blur,add_blur_isotropic

def addGaussNoise(s):
    #var = random.uniform(0.0001, 1.0)
    var = 0.25
    noisy = skimage.util.random_noise(s, mode='gaussian', var=var)
    return noisy



sub_dataset_list = ['/data1/liguanlin/Datasets/iHarmony/HFlickr','/data1/liguanlin/Datasets/iHarmony/HCOCO','/data1/liguanlin/Datasets/iHarmony/HAdobe5k']

for sub_dataset_name in sorted(sub_dataset_list):
    
    comp_file_path = sub_dataset_name + '/composite_images/' #原始合成图像路径
    comp_file_list = os.listdir(comp_file_path) #获取合成图像的list

    mask_file_path = sub_dataset_name + '/masks/' #原始合成图像mask路径

    for image in sorted(comp_file_list)[35:]:
        img_path = comp_file_path + image #原始合成图像地址
        print(img_path)
        comp = Image.open(img_path).convert('RGB') #读取合成图像
        numpy_comp = np.array(comp) #将合成图像转换为numpy格式

        name_parts=img_path.split('_')
        mask_img_path = mask_file_path + image 

        mask_path = mask_img_path.replace(('_'+name_parts[-1]),'.png')#mask地址

        mask = Image.open(mask_path).convert('1')  #读取mask图像
        numpy_mask = np.array(mask) #将mask转换为numpy格式

        width = numpy_comp.shape[0]
        height = numpy_comp.shape[1]

        noisy_comp = addGaussNoise(numpy_comp.view()) #添加噪声，生成全图噪声
        numpy_noisy_comp = np.array(noisy_comp*255, dtype='uint8') #将噪声图转换numpy格式

        blur_img = add_blur_isotropic(img_path)
        #生成一张空图，用来接受原始合成图的背景和噪声全图的前景
        noisy_roi = np.zeros((numpy_comp.shape[0], numpy_comp.shape[1], numpy_comp.shape[2]), dtype='uint8')

        #给空图赋值，将原始合成图的背景和噪声全图的前景合并起来
        for i in range(width):
            for j in range(height):
                if numpy_mask[i,j] == False:
                    noisy_roi[i,j,:] = blur_img[i,j,:] #原始合成图像背景赋值给背景
                elif numpy_mask[i,j] == True:
                    noisy_roi[i,j,:] = numpy_noisy_comp[i,j,:] #噪声图像背景赋值给背景

        #生成带噪图的存储文件夹
        save_noisy_img_path = sub_dataset_name + '/composite_noisy25_f_blured_b_images_istropic/'
        if not os.path.exists(save_noisy_img_path):
            os.mkdir(save_noisy_img_path)
        noisy_img_name = save_noisy_img_path + image #带噪图的image地址

        save_img = Image.fromarray(noisy_roi)#从numpy数组生成图像
        save_img.save(noisy_img_name)
