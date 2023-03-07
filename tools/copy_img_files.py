import os

from shutil import copyfile

"""
dataset_root = '/data1/liguanlin/Datasets/iHarmony/Hday2night/'
trainfile = dataset_root+'Hday2night_train.txt' #修改点1，替换HCOCO_train.txt
train_dst_root = '/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_train_without_noise/'
with open(trainfile,'r') as f:
        for line in f.readlines():
            img_path = os.path.join(dataset_root, 'composite_images', line.rstrip())
            dst_img_path = os.path.join(train_dst_root, line.rstrip())
            copyfile(img_path, dst_img_path)

testfile = dataset_root+'Hday2night_test.txt' #修改点1，替换HCOCO_train.txt
test_dst_root = '/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_test_without_noise/'
with open(testfile,'r') as f:
        for line in f.readlines():
            img_path = os.path.join(dataset_root, 'composite_images', line.rstrip())
            dst_img_path = os.path.join(test_dst_root, line.rstrip())
            copyfile(img_path, dst_img_path)
"""
dataset_root = '/data1/liguanlin/Datasets/iHarmony/Hday2night/'
trainfile = dataset_root+'Hday2night_train.txt' #修改点1，替换HCOCO_train.txt
train_dst_root = '/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_train_downsampled/'
with open(trainfile,'r') as f:
        for line in f.readlines():
            img_path = os.path.join(dataset_root, 'composite_downsampled_images', line.rstrip())
            dst_img_path = os.path.join(train_dst_root, line.rstrip())
            copyfile(img_path, dst_img_path)

testfile = dataset_root+'Hday2night_test.txt' #修改点1，替换HCOCO_train.txt
test_dst_root = '/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_test_downsampled/'
with open(testfile,'r') as f:
        for line in f.readlines():
            img_path = os.path.join(dataset_root, 'composite_downsampled_images', line.rstrip())
            dst_img_path = os.path.join(test_dst_root, line.rstrip())
            copyfile(img_path, dst_img_path)