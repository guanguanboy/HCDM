import os

from shutil import copyfile


restored_img_root = '/data1/liguanlin/research_projects/DPM/Palette-Image-to-Image-Diffusion-Models/experiments/test_restoration_noise_day2night_220514_110556/results/test/0/'

output_prefix = 'Out_'
all_img_list = os.listdir(restored_img_root)
restored_img_list_len = len(all_img_list)
print('total img list len=', len(all_img_list))

print(all_img_list[0][0:4])
restored_img_list = []

dst_root = '/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_test_restored/'

for i in range(restored_img_list_len):
    if all_img_list[i][0:4] == output_prefix:
        restored_img_list.append(all_img_list[i])

        original_img_path = restored_img_root + all_img_list[i]
        dst_img_name = all_img_list[i][4:]
        dst_img_path = dst_root + dst_img_name

        copyfile(original_img_path, dst_img_path)

print('restored img list len=', len(restored_img_list))




