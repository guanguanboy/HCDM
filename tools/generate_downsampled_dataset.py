import numpy as np
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import skimage
import random

def addGaussNoise(s):
    var = random.uniform(0.0001, 0.25)
    noisy = skimage.util.random_noise(s, mode='gaussian', var=var)
    return noisy
sub_dataset_list = ['/data1/liguanlin/Datasets/iHarmony/Hday2night']

for sub_dataset_name in sub_dataset_list:
    
    comp_file_path = sub_dataset_name + '/composite_images/'

    comp_file_list = os.listdir(comp_file_path) #合成图像列表

    mask_file_path = sub_dataset_name + '/masks/' #掩码图像保存路径

    for image in comp_file_list:
        img_path = comp_file_path + image
        #print(img_path)
        comp = Image.open(img_path).convert('RGB')
        #print('comp.size=', comp.size)
        numpy_comp = np.array(comp)
        #print('numpy_comp.shape=', numpy_comp.shape)
        name_parts=img_path.split('_')
        mask_img_path = mask_file_path + image

        mask_path = mask_img_path.replace(('_'+name_parts[-1]),'.png')

        mask = Image.open(mask_path).convert('1') #获取到掩码图像
        numpy_mask = np.array(mask) #将掩码图像转换为numpy格式

        #noisy_comp = addGaussNoise(numpy_comp.view()) #给合成图像添加噪声


        #numpy_noisy_comp = np.array(noisy_comp*255, dtype='uint8') #将带噪合成图像转换为uint8型

        origin_img_width = numpy_comp.shape[1]
        origin_img_height = numpy_comp.shape[0]
        origin_channel = numpy_comp.shape[2]
        origin_sizes=(origin_img_width, origin_img_height)

        new_img_width = origin_img_width//8
        new_img_height = origin_img_height//8

        dst_sizes=(new_img_width, new_img_height)
        #step1：先把整个图像downsample。
        downsampled_img = comp.resize(dst_sizes, Image.BICUBIC)

        #step2：然后再upsample。
        up_sampled_img = downsampled_img.resize(origin_sizes, Image.BICUBIC)

        up_sampled_img_numpy = np.array(up_sampled_img)
        #print('up_sampled_img_numpy.shape=', up_sampled_img_numpy.shape)  #(461, 639, 3)

        #step3：然后将mask区域复制过来。
        output_img = np.zeros((origin_img_height, origin_img_width, origin_channel), dtype='uint8')
        #print('output_img.shape=',output_img.shape)

        for i in range(origin_img_height):
            for j in range(origin_img_width):
                if numpy_mask[i,j] == False:
                    #print('True')
                    output_img[i,j,:] = numpy_comp[i,j,:] #从原始合成图中选择像素
                elif numpy_mask[i,j] == True:
                    output_img[i,j,:] = up_sampled_img_numpy[i,j,:] #从带噪合成图像中选择像素

                    #print('False')
                    #continue

        #指定图像保存的路径
        save_downsampled_img_path = sub_dataset_name + '/composite_downsampled_images_8x/'
        if not os.path.exists(save_downsampled_img_path):
            os.mkdir(save_downsampled_img_path)
        downsampled_img_name = save_downsampled_img_path + image

        save_img = Image.fromarray(output_img)#保存最终的图像
        print('save_img.size=', save_img.size)
        save_img.save(downsampled_img_name)
