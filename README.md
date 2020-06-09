# Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining 
This repository is for CSNLN introduced in the following paper

[Yiqun Mei](http://yiqunm2.web.illinois.edu/), [Yuchen Fan](https://scholar.google.com/citations?user=BlfdYL0AAAAJ&hl=en), [Yuqian Zhou](https://yzhouas.github.io/), [Lichao Huang](https://scholar.google.com/citations?user=F2e_jZMAAAAJ&hl=en), [Thomas S. Huang](https://ifp-uiuc.github.io/), and [Humphrey Shi](https://www.humphreyshi.com/), "Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining", CVPR2020, [[arXiv]](https://arxiv.org/abs/2006.01424) 


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 18.04 environment (Python3.6, PyTorch_1.1.0) with Titan /Xp, V100 GPUs. 
## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction

Deep convolution-based single image super-resolution (SISR) networks embrace the benefits of learning from large-scale external image resources for local recovery, yet most existing works have ignored the long-range feature-wise similarities in natural images. Some recent works have successfully leveraged this intrinsic feature correlation by exploring non-local attention modules. However, none of the current deep models have studied another inherent property of images: cross-scale feature correlation. In this paper, we propose the first Cross-Scale Non-Local (CS-NL) attention module with integration into a recurrent neural network. By combining the new CS-NL prior with local and in-scale non-local priors in a powerful recurrent fusion cell, we can find more cross-scale feature correlations within a single low-resolution (LR) image. The performance of SISR is significantly improved by exhaustively integrating all possible priors. Extensive experiments demonstrate the effectiveness of the proposed CS-NL module by setting new state-of-the-arts on multiple SISR benchmarks.

![CS-NL Attention](/Figs/Attention.png)

Cross-Scale Non-Local Attention.

![CSNLN](/Figs/CSNLN.png)

The recurrent architecture with Self-Exemplars Mining (SEM) Cell.

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. 

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. (optional) Download pretrained models for our paper.

    All the models and visual results can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1C--KFcfAsvLM4K0J6td4TqNeiaGsTXd_?usp=sharing)

2. Cd to 'src', run the following script to train models.

    **Example command is in the file 'demo.sh'.**

    ```bash
    # Example X2 SR
    python3 main.py --chop --batch_size 16 --model CSNLN --scale 2 --patch_size 96 --save CSNLN_x2 --n_feats 128 --depth 12 --data_train DIV2K --save_models

    ```

## Test
### Quick start
1. Download benchmark datasets from [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

1. (optional) Download pretrained models for our paper.

    All the models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1C--KFcfAsvLM4K0J6td4TqNeiaGsTXd_?usp=sharing)

2. Cd to 'src', run the following scripts.

    **Example command is in the file 'demo.sh'.**

    ```bash
    # No self-ensemble: CSNLN
    # Example X2 SR
    python3 main.py --model CSNLN --data_test Set5+Set14+B100+Urban100 --data_range 801-900 --scale 2 --n_feats 128 --depth 12 --pre_train ../models/model_x2.pt --save_results --test_only --chop
    ```

## Results
### Quantitative Results
![PSNR_SSIM](/Figs/PSNR_SSIM.png)

For more results, please refer to our [paper](https://arxiv.org/abs/2006.01424) and [Supplementary Materials](http://yiqunm2.web.illinois.edu/assets/papers/04292-supp.pdf).
### Visual Results
![Visual_PSNR_SSIM](/Figs/Visual_1.png)

![Visual_PSNR_SSIM](/Figs/Visual_2.png)

![Visual_PSNR_SSIM](/Figs/Visual_3.png)


## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@inproceedings{Mei2020image,
  title={Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining},
  author={Mei, Yiqun and Fan, Yuchen and Zhou, Yuqian and Huang, Lichao and Huang, Thomas S and Shi, Humphrey},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}

```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [generative-inpainting-pytorch](https://github.com/daa233/generative-inpainting-pytorch). We thank the authors for sharing their codes.

