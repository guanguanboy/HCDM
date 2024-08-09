This repo provides the official  code  and datasets of the paper 'Image Harmonization in Complex Degradation Scenes'

### D-iHarmony4 dataset

**We release the D-iHarmony4 dataset**. It contains 4 sub-datasets: **D-HCOCO**,**D-HAdobe5k**, **D-HFlickr**, and **D-Hday2night**, each of which contains degraded composite images, foreground masks of composite images and corresponding real images. The D-iHarmony4 dataset is provided in  [**Baidu Cloud**](https://pan.baidu.com/s/1z6VfVOKCJiQxTDxnVcsVJA) (access code: 5pfl).

| |D-HCOCO|D-HAdobe5k|D-HFlickr|D-Hday2night|D-iHarmony4 |
|:--:|:--:|:--:|:--:|:--:|:--:|
|Training set| 38545 |19437| 7449 |311|65742|
|Test set| 4283 |2160| 828 |133|7404|


### 1. D-HCOCO

D-HCOCO, containing 42k degraded composite images, is generated based on [Microsoft COCO](<http://cocodataset.org/>) dataset. The foreground region is corresponding object segmentation mask provided from COCO. Within the foreground region, the appearance of COCO image is edited using various color transfer methods. 


### 2. D-HAdobe5k

D-HAdobe5k is generated based on [MIT-Adobe FiveK](<http://data.csail.mit.edu/graphics/fivek/>) dataset. Provided with 6 editions of the same image, we manually segment the foreground region and exchange foregrounds between 2 versions. 


### 3. D-HFlickr

We collected 4833 images from [Flickr](<https://www.flickr.com/>). After manually segmenting the foreground region, we use the same method as D-HCOCO to generate HFlickr sub-dataset. 


### 4. D-Hday2night

D-Hday2night is generated based on [day2night](https://pan.baidu.com/s/1bCtVhhtb_EDool_UnN2Bjw) dataset. We manually segment the foreground region, which is cropped and overlaid on another image captured on a different time. 

### The Original iHarmony4 dataset

The original iHarmony4 dataset is provided in  [**Baidu Cloud**](https://pan.baidu.com/s/1xEN0Xrv_MbuKT0ZqsipeEg) (access code: kqz3) and [**One Drive**](https://1drv.ms/f/s!AohNSvvkuxZmgTHOraRzo5-X3nMp?e=bQQKkR).

### Pre-trained Models

* The pretrained models are  at [Baidu Cloud](https://pan.baidu.com/s/1uUe7u-oW-iPuZI-d3jKsfA) (access code：qeek) and [google drive](https://drive.google.com/drive/folders/1SSojkgJgUM41jzwR9xbN6ki2jFqsr33E?usp=sharing)


### Pre-requirements

```
pip install -r requirements.txt
```

### Train

train on day2night dataset:

```
python -W ignore run.py -p train -c config/harmonization_day2night_modified_allinone.json
```


### Test

2D Map:

Download the pretained models frist and put the models under 'pretrained_model\checkpoint\2d_map\ ' directory. Then run the follow code:

```
python -W ignore run.py -p test -c config/harmonization_day2night_modified_allinone_2d_test.json
```

1D Embedding:

Download the pretained models frist and put the models under 'pretrained_model\checkpoint\1D_embed\' directory. Then run the follow code:

```
python -W ignore run.py -p test -c config/harmonization_day2night_modified_allinone.json
```
The results will be gengerated in a directory like this:
```
experiments/test_harmonization_allinone_220818_115348/results/test/0
```
### Evaluation

Change the output_path variable in 'evaluate.py' to the complete generated result directory.

```
experiments/test_harmonization_allinone_220818_115348/results/test/0
```
Then run the fllowing code:

```
python evaluation/evaluate.py
```

The evalution results will be displayed on console as follows:
```
MSE 37.73 | PSNR 36.89 | SSIM 0.975 |fMSE 643.64 | fPSNR 21.79 | fSSIM 0.5056
```
## **Contact**
Please contact me if there is any question (guanguanboy@gmail.com)
