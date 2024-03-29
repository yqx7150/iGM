# iGM

**Paper**: Progressive Colorization via Iterative Generative Models

**Authors**: Jinjie Zhou, Kai Hong, Tao Deng, Qiegen Liu, IEEE Signal Processing Letters, vol. 27, pp. 2054-2058, 2020.  
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
## Graphical representation

<div align="center"><img src="https://github.com/yqx7150/iGM//blob/master/1222.png">  </div>
The interpretation of the generalized ideology behind iGM. From left to right:CVAE,iGM 


<div align="center"><img src="https://github.com/yqx7150/iGM/blob/master/11111.png"> </div>
Four colorized images on LUSN-bedroom and LUSN-church dataset. From left to right:Ground truth, Gray input, Zhang et.al , CGAN, Memopainter, ChromaGAN, iGM-3C, iGM-6C


### Other Related Projects

  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9435335)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

 * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
   
  * Universal Generative Modeling for Calibration-free Parallel MR Imaging  
[<font size=5>**[Paper]**</font>](https://biomedicalimaging.org/2022/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UGM-PI)   [<font size=5>**[Poster]**</font>](https://github.com/yqx7150/UGM-PI/blob/main/paper%20%23160-Poster.pdf)        
    
 * Joint Intensity-Gradient Guided Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2012.14130)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/JGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

 * Wavelet Transform-assisted Adaptive Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2107.04261)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WACM)   [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

 * Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT) 
