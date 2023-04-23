import numpy as np
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import skimage
import random


sub_dataset_list = ['/data1/liguanlin/Datasets/iHarmony/HAdobe5k']

for sub_dataset_name in sub_dataset_list:
    
    comp_file_path = sub_dataset_name + '/composite_noisy25_images/'

    comp_file_list = os.listdir(comp_file_path) #合成图像列表

    for image in comp_file_list:
        img_path = comp_file_path + image
        #print(img_path)
        comp = Image.open(img_path).convert('RGB')
        #print('comp.size=', comp.size)

        new_img_width = 1024
        new_img_height = 1024

        dst_sizes=(new_img_width, new_img_height)
        #step1：先把整个图像downsample。
        downsampled_img = comp.resize(dst_sizes, Image.BICUBIC)


        #指定图像保存的路径
        save_downsampled_img_path = '/data1/liguanlin/Datasets/iHarmony/HAdobe5k_1024/composite_noisy25_images/'
        if not os.path.exists(save_downsampled_img_path):
            os.mkdir(save_downsampled_img_path)
        downsampled_img_name = save_downsampled_img_path + image

        print('save_img.size=', downsampled_img.size)
        downsampled_img.save(downsampled_img_name)
