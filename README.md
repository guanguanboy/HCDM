This repo contains the code  paper:

Image Harmonization in Complex Degradation Scenes

### Pre-trained Models

* The pretrained models are in [Baidu Cloud](https://pan.baidu.com/s/1uUe7u-oW-iPuZI-d3jKsfA) (access code：qeek).
* [google drive](https://drive.google.com/drive/folders/1SSojkgJgUM41jzwR9xbN6ki2jFqsr33E?usp=sharing)

> Other Pre-trained Models are still reorganizing and uploading, it will be released soon.


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
