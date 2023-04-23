import os

from shutil import copyfile


dataset_root = '/data1/liguanlin/Datasets/iHarmony/HAdobe5k_1024/'
trainfile = dataset_root +'HAdobe5k_1024_train.txt' #修改点1，替换HCOCO_train.txt
train_dst_root = '/data1/liguanlin/Datasets/iHarmony/composite_images_train_noisy25/'
train_count = 0
test_count = 0
with open(trainfile,'r') as f:
        for line in f.readlines():
            img_path = os.path.join(dataset_root, 'composite_noisy25_images', line.rstrip())
            line_strip = line.rstrip()
            dst_img_path = os.path.join(train_dst_root, line_strip)
            copyfile(img_path, dst_img_path)
            train_count = train_count + 1
print('train_count==', train_count)

testfile = dataset_root+'HAdobe5k_1024_test.txt' #修改点1，替换HCOCO_train.txt
test_dst_root = '/data1/liguanlin/Datasets/iHarmony/composite_images_test_noisy25/'
with open(testfile,'r') as f:
        for line in f.readlines():
            img_path = os.path.join(dataset_root, 'composite_noisy25_images', line.rstrip())
            line_strip = line.rstrip()
            dst_img_path = os.path.join(test_dst_root, line_strip)
            copyfile(img_path, dst_img_path)
            test_count = test_count + 1
print('test_count==', test_count)            

mask_count = 0
mask_root = dataset_root + 'masks/'