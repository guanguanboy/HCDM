import argparse
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as tf
import pytorch_ssim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #指定第一块gpu

if __name__ == '__main__':

    real_paths = []
    mask_paths = []
    composite_paths = []
    composite_img_path_name = 'composite_noisy25_images'
    #composite_img_path_name = 'composite_images' #计算原数据集的Input composite。
    subdataset_path = '/data1/liguanlin/Datasets/iHarmony/HCOCO/'

    #files = '/data1/liguanlin/Datasets/iHarmony/Hday2night/Hday2night_test.txt'
    #files = '/data1/liguanlin/Datasets/iHarmony/HFlickr/HFlickr_test.txt'
    files = '/data1/liguanlin/Datasets/iHarmony/HCOCO/HCOCO_test.txt'
    #files = '/data1/liguanlin/Datasets/iHarmony/HAdobe5k/HAdobe5k_test.txt'
    with open(files,'r') as f:
            for line in f.readlines():
                name_str = line.rstrip()
                composite_img_file_name = os.path.join(subdataset_path, composite_img_path_name, name_str)
                print(name_str)
                print(composite_img_file_name)

                name_parts=name_str.split('_')

                target_path = composite_img_file_name.replace(composite_img_path_name,'real_images') # #修改点6，如果是带噪声的训练，将这里修改为composite_noisy25_images
                target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')                

                mask_img_name = name_str.replace(('_'+name_parts[-1]),'.png')

                mask_path = os.path.join(subdataset_path,'masks' ,mask_img_name)

                real_paths.append(target_path)
                mask_paths.append(mask_path)
                composite_paths.append(composite_img_file_name)

    mse_scores = 0
    psnr_scores = 0
    ssim_scores = 0
    image_size = 256
    fmse_scores = 0
    fpsnr_scores = 0
    fssim_scores = 0

    fore_area_count = 0
    fmse_score_list = []

    count = 0
    ssim_window_default_size = 11
    for i, composite_path in enumerate(tqdm(composite_paths)):
        count += 1

        composite = Image.open(composite_path).convert('RGB')
        newsize = (256, 256)
        composite = composite.resize(newsize, Image.BICUBIC)

        real = Image.open(real_paths[i]).convert('RGB')
        real = real.resize(newsize, Image.BICUBIC)

        composite_np = np.array(composite, dtype=np.float32)
        real_np = np.array(real, dtype=np.float32)

        mse_score = mse(composite_np, real_np)
        psnr_score = psnr(real_np, composite_np, data_range=255)
        ssim_score = ssim(real_np, composite_np, data_range=255, multichannel=True)

        psnr_scores += psnr_score
        mse_scores += mse_score
        ssim_scores += ssim_score

        """下面分别计算fmse, fpsnr和fssim"""
        mask = Image.open(mask_paths[i]).convert('1') #获取mask区域。
        mask = tf.resize(mask, [image_size,image_size], interpolation=Image.BICUBIC)

        mask = tf.to_tensor(mask).unsqueeze(0).cuda()
        composite = tf.to_tensor(composite_np).unsqueeze(0).cuda()
        real = tf.to_tensor(real_np).unsqueeze(0).cuda()

        fore_area = torch.sum(mask)
        fmse_score = torch.nn.functional.mse_loss(composite*mask,real*mask)*256*256/fore_area #计算得到fmse        
        fmse_score = fmse_score.item()

        fpsnr_score = 10 * np.log10((255 ** 2) / fmse_score) #计算得到fpsnr
        
        ssim_score, fssim_score = pytorch_ssim.ssim(composite, real, window_size=ssim_window_default_size, mask=mask) #计算得到fssim
        fmse_scores += fmse_score
        fpsnr_scores += fpsnr_score
        fssim_scores += fssim_score


    mse_scores_mu = mse_scores/count
    psnr_scores_mu = psnr_scores/count
    ssim_scores_mu = ssim_scores/count
    fpsnr_scores_mu = fpsnr_scores/count
    fmse_scores_mu = fmse_scores/count
    fssim_score_mu = fssim_scores/count

    print(count)
    mean_sore = "MSE %0.2f | PSNR %0.2f | SSIM %0.3f |fMSE %0.2f | fPSNR %0.2f | fSSIM %0.4f" % (mse_scores_mu, psnr_scores_mu, ssim_scores_mu,fmse_scores_mu,fpsnr_scores_mu,fssim_score_mu)
    print(mean_sore)    