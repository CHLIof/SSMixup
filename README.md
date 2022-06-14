# SSMixup:Simple Style transfer Mixing Data Augmentation

Implementation of [SSMixup:Simple Style transfer Mixing Data Augmentation].

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# train on CIFAR-10: 
python SSMixup.py -d cifar-10 -m resnet-18 --r 0.8 --alpha 0.5

python SSCutMix.py -d cifar-10 -m resnet-18 --prob 0.5 --r 0.7 --alpha2 0.8

python SSCutout.py -d cifar-10 -m resnet-18 --length 16 --alpha 0.5

# train on CIFAR-100: 
python SSMixup.py -d cifar-100 -m resnet-18 --r 0.8 --alpha 0.5

python SSCutMix.py -d cifar-100 -m resnet-18 --prob 0.5 --r 0.8 --alpha2 0.8

python SSCutout.py -d cifar-100 -m resnet-18 --length 16 --alpha 0.5
```
## Acknowledgement
The implementation of SSMixup is adapted from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repository by [kuangliu](https://github.com/kuangliu).


## Citation
```
@InProceedings{hong2021stylemix,
    author    = {Minui Hong and Jinwoo Choi and Gunhee Kim},
    title     = {StyleMix: Separating Content and Style for Enhanced Data Augmentation},
    booktitle = {CVPR},
    year      = {2021}
}
```
```
@article{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
```
```
@inproceedings{yun2019cutmix,
    title={CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features},
    author={Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Chun, Sanghyuk and Choe, Junsuk and Yoo, Youngjoon},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year={2019},
    pubstate={published},
    tppubtype={inproceedings}
}
```
```
@article{devries2017cutout,  
  title={Improved Regularization of Convolutional Neural Networks with Cutout},  
  author={DeVries, Terrance and Taylor, Graham W},  
  journal={arXiv preprint arXiv:1708.04552},  
  year={2017}  
}
```