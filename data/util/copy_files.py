import os
from shutil import copyfile

"""
image_origin_path = 'home/cj/origin_img/1.jpg'
image_target_path = 'home/cj/target_img/2.jpg'

"""
def images_copy(image_origin_path, image_target_path):
    if os.path.exists(image_target_path) or not (os.path.exists(image_origin_path)):
        print('%s is exited' % image_origin_path)
    else:
        copyfile(image_origin_path, image_target_path)
        print('copy %s is done ' % image_target_path)

image_paths = []
data_root = '/data1/liguanlin/Datasets/iHarmony/Hday2night/'
trainfile =data_root+'Hday2night_train.txt' #修改点1，替换HCOCO_train.txt
with open(trainfile,'r') as f:
        for line in f.readlines():
            image_paths.append(os.path.join(data_root, 'composite_noisy25_images', line.rstrip())) #修改点2，增加composite_images，如果是带噪声的训练，将这里修改为composite_noisy25_images

for index in range(len(image_paths)):
    target_image_path =  image_paths[index].replace('composite_noisy25_images','composite_images_train') # #修改点5，如果是带噪声的训练，将这里修改为composite_noisy25_images

    images_copy(image_paths[index], target_image_path)

image_paths_test = []

test_file =data_root+'Hday2night_test.txt' #修改点1，替换HCOCO_train.txt
with open(test_file,'r') as f:
        for line in f.readlines():
            image_paths_test.append(os.path.join(data_root, 'composite_noisy25_images', line.rstrip())) #修改点2，增加composite_images，如果是带噪声的训练，将这里修改为composite_noisy25_images

for index in range(len(image_paths_test)):
    target_image_path_test =  image_paths_test[index].replace('composite_noisy25_images','composite_images_test') # #修改点5，如果是带噪声的训练，将这里修改为composite_noisy25_images

    images_copy(image_paths_test[index], target_image_path_test)