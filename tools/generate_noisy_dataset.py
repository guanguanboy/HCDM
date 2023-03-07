import numpy as np
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import skimage
import random

def addGaussNoise(s):
    #var = random.uniform(0.0001, 1.0)
    var = 0.75
    noisy = skimage.util.random_noise(s, mode='gaussian', var=var)
    return noisy
sub_dataset_list = ['/data1/liguanlin/Datasets/iHarmony/HAdobe5k']

for sub_dataset_name in sorted(sub_dataset_list):
    
    comp_file_path = sub_dataset_name + '/composite_images/'

    comp_file_list = os.listdir(comp_file_path)

    mask_file_path = sub_dataset_name + '/masks/'

    for image in sorted(comp_file_list)[35:]:
        img_path = comp_file_path + image
        print(img_path)
        comp = Image.open(img_path).convert('RGB')
        numpy_comp = np.array(comp)

        name_parts=img_path.split('_')
        mask_img_path = mask_file_path + image

        mask_path = mask_img_path.replace(('_'+name_parts[-1]),'.png')

        mask = Image.open(mask_path).convert('1')
        numpy_mask = np.array(mask)

        width = numpy_comp.shape[0]
        height = numpy_comp.shape[1]

        noisy_comp = addGaussNoise(numpy_comp.view())
        #noisy_comp = np.random.randn(width, height, numpy_comp.shape[2])

        numpy_noisy_comp = np.array(noisy_comp*255, dtype='uint8')


        noisy_roi = np.zeros((numpy_comp.shape[0], numpy_comp.shape[1], numpy_comp.shape[2]), dtype='uint8')
        #noisy_roi = (np.random.normal(0, 0.25, (numpy_comp.shape[0], numpy_comp.shape[1], numpy_comp.shape[2]))*255).astype(np.uint8)

        for i in range(width):
            for j in range(height):
                if numpy_mask[i,j] == False:
                    #print('True')
                    noisy_roi[i,j,:] = numpy_comp[i,j,:]
                    #noisy_roi[i,j,:] = 0
                elif numpy_mask[i,j] == True:
                    noisy_roi[i,j,:] = numpy_noisy_comp[i,j,:]
                    #noisy_roi[i,j,:] = noisy_roi[i,j,:]
                    #print('False')
                    #continue

        save_noisy_img_path = sub_dataset_name + '/composite_noisy75_images/'
        if not os.path.exists(save_noisy_img_path):
            os.mkdir(save_noisy_img_path)
        noisy_img_name = save_noisy_img_path + image
        save_img = Image.fromarray(noisy_roi)
        save_img.save(noisy_img_name)
