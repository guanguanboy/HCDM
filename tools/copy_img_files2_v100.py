import os

from shutil import copyfile


dataset_root = '/mnt/cfs/liguanlin/Datasets/iHarmony4/'
trainfile = dataset_root +'IHD_noisy25_train.txt' #修改点1，替换HCOCO_train.txt
train_dst_root = '/mnt/cfs/liguanlin/Datasets/iHarmony4/composite_images_train/'
with open(trainfile,'r') as f:
        for line in f.readlines():
            img_path = os.path.join(dataset_root, line.rstrip())
            line_strip = line.rstrip()
            path_list = line_strip.split('/')
            dst_img_path = os.path.join(train_dst_root, path_list[-1])
            copyfile(img_path, dst_img_path)

testfile = dataset_root+'IHD_noisy25_test.txt' #修改点1，替换HCOCO_train.txt
test_dst_root = '/mnt/cfs/liguanlin/Datasets/iHarmony4/composite_images_test/'
with open(testfile,'r') as f:
        for line in f.readlines():
            img_path = os.path.join(dataset_root, line.rstrip())
            line_strip = line.rstrip()
            path_list = line_strip.split('/')
            dst_img_path = os.path.join(test_dst_root, path_list[-1])
            copyfile(img_path, dst_img_path)