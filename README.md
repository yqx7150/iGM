# iGM

**Paper**: Progressive Colorization via Interative Generative Models

**Authors**: Jinjie Zhou, Kai Hong, Tao Deng, Qiegen Liu

Published in IEEE Signal Processing Letters  
https://ieeexplore.ieee.org/document/9258392

Date : 9/2020
Version : 1.0   
The code and the algorithm are for non-comercial use only. 
Copyright 2020, Department of Electronic Information Engineering, Nanchang University.  

Colorization is the process of coloring monochrome images. It has been widely used in photo processing and scientific illustration. However, colorizing grayscale images is an intrinsic ill-posed and ambiguous problem, with multiple plausible solutions. To address this issue, we develop a nov-el progressive automatic colorization via iterative genera-tive models (iGM) that can produce satisfactory coloriza-tion in an unsupervised manner. In particular, we convert colors flexibly in multi-color spaces which is regarding as the key prior information to pave the way for producing the most probable colorization in high dimensional space. Experiments on indoor and outdoor scenes reveal that iGM produces more realistic and finer results, compared to state-of-the-arts.

## Training
if you want to train the code, please train the 3 channel model

```bash 
python3 iGM_main.py --igm anneal_runner_train_3ch --config anneal.yml --doc your save path
```

Then

```bash
python3 iGM_main.py --igm anneal_runner_train_6ch --config anneal.yml --doc your save path
```

## Test
if you want to test the code, please test the 3 channel model

```bash 
python3 iGM_main.py --igm Test_3ch --config anneal.yml --doc your checkpoint --test --image_folder your save path
```
Then

```bash
python3 iGM_main.py --igm Test_6ch --config anneal.yml --doc your checkpoint --test --image_folder your save path
```


## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Baidu Drive](https://pan.baidu.com/s/1do-Y-13E7NWK2mkE9K912w). 
key number is "IGM6 " 

## Test Data
We randomly select 100 bedrooms and church data respectively, the size is 128x128.
