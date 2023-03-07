from torchvision.transforms import functional as trans_fn 
from PIL import Image
import numpy as np

img_path = './test_imgs/d19000-99.jpg'
img = Image.open(img_path).convert('RGB')
img_numpy = np.array(img)
print(img_numpy.shape)
origin_img_width = img_numpy.shape[0]
origin_img_height = img_numpy.shape[1]
origin_sizes=(origin_img_width, origin_img_height)

new_img_width = origin_img_width//2
new_img_height = origin_img_height//2

dst_sizes=(new_img_width, new_img_height)

#trans_fn.resize函数本质上使用的还是Image.resize函数
#downsampled_img = trans_fn.resize(img, dst_sizes, Image.BICUBIC) #img should be PIL Image
downsampled_img = img.resize(dst_sizes, Image.BICUBIC)
saved_downsampled_img_path = './test_imgs/d19000-99_downsampled.jpg'
downsampled_img.save(saved_downsampled_img_path)

#up_sampled_img = trans_fn.resize(downsampled_img, origin_sizes, Image.BICUBIC)
up_sampled_img = downsampled_img.resize(origin_sizes, Image.BICUBIC)
save_img = up_sampled_img

saved_img_path = './test_imgs/d19000-99_bicubic.jpg'

save_img.save(saved_img_path)

