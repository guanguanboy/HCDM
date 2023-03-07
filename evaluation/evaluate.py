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

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #指定第一块gpu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, help='Generate images directory')

    args = parser.parse_args()

    harmonized_paths = []
    real_paths = []
    mask_paths = []

    #output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_220505_145224/results/test/0/'
    #output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_220505_162755/results/test/0/'
    #output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_220512_113835/results/test/0/'
    #output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_220512_155132/results/test/0/'
    #output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_220514_102412/results/test/0/'
    #output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_220514_150437/results/test/0/'
    #output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_220527_165832/results/test/0/'
    output_path = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_harmonization_day2night_improved_220803_153028/results/test/0/'




    files = '/data1/liguanlin/Datasets/iHarmony/Hday2night/Hday2night_test.txt'
    #files = '/mnt/cfs/liguanlin/Datasets/iHarmony4/HFlickr/HFlickr_test.txt'
    #files = '/mnt/cfs/liguanlin/Datasets/iHarmony4/HCOCO/HCOCO_test.txt'
    #files = '/mnt/cfs/liguanlin/Datasets/iHarmony4/HAdobe5k/HAdobe5k_test.txt'
    with open(files,'r') as f:
            for line in f.readlines():
                name_str = line.rstrip()
                
                harmonized_img_name = 'Out_' + name_str
                harmonized_path = os.path.join(output_path, harmonized_img_name)
                
                real_img_name = 'In_' + name_str
                real_path = os.path.join(output_path, real_img_name)

                name_parts=name_str.split('_')
                mask_img_name = name_str.replace(('_'+name_parts[-1]),'.png')

                mask_path = '/data1/liguanlin/Datasets/iHarmony/Hday2night/masks/' + mask_img_name

                real_paths.append(real_path)
                harmonized_paths.append(harmonized_path)
                mask_paths.append(mask_path)

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
    for i, harmonized_path in enumerate(tqdm(harmonized_paths)):
        count += 1

        harmonized = Image.open(harmonized_path).convert('RGB')
        real = Image.open(real_paths[i]).convert('RGB')

        harmonized_np = np.array(harmonized, dtype=np.float32)
        real_np = np.array(real, dtype=np.float32)

        mse_score = mse(harmonized_np, real_np)
        psnr_score = psnr(real_np, harmonized_np, data_range=255)
        ssim_score = ssim(real_np, harmonized_np, data_range=255, multichannel=True)

        psnr_scores += psnr_score
        mse_scores += mse_score
        ssim_scores += ssim_score

        """下面分别计算fmse, fpsnr和fssim"""
        mask = Image.open(mask_paths[i]).convert('1') #获取mask区域。
        mask = tf.resize(mask, [image_size,image_size], interpolation=Image.BICUBIC)

        mask = tf.to_tensor(mask).unsqueeze(0).cuda()
        harmonized = tf.to_tensor(harmonized_np).unsqueeze(0).cuda()
        real = tf.to_tensor(real_np).unsqueeze(0).cuda()

        fore_area = torch.sum(mask)
        fmse_score = torch.nn.functional.mse_loss(harmonized*mask,real*mask)*256*256/fore_area #计算得到fmse        
        fmse_score = fmse_score.item()

        fpsnr_score = 10 * np.log10((255 ** 2) / fmse_score) #计算得到fpsnr
        
        ssim_score, fssim_score = pytorch_ssim.ssim(harmonized, real, window_size=ssim_window_default_size, mask=mask) #计算得到fssim
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