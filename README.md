# GAIA: Delving into Gradient-based Attribution Abnormality for Out-of-distribution Detection

Code for NeurIPS2023 paper "**GAIA: Delving into Gradient-based Attribution Abnormality for Out-of-distribution Detection**" .

[Paper](https://arxiv.org/pdf/2311.09620.pdf)

## Preparation

To prepare data for CIFAR benchmarks, `CIFAR10`, `CIFAR100` and `SVHN` are directly downloaded from torchvision. For  `TinyImagenet`, `LSUN`, `Textures` and `Places`, please download from 

```
https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
https://www.robots.ox.ac.uk/~vgg/data/dtd/  
http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

Then put them in one dir, like

```
- datasets
    - LSUN
    - Imagenet
    - Places
    - textures
       - images
       - ...
    ...
```

To prepare models for CIFAR benchmarks, please download from 
```
https://drive.google.com/drive/u/0/folders/12i4Kq97h0ngJt9V6IpfgUPvFXDEznf5J
```

To prepare data for ImageNet benchmark, please follow [GradNorm](https://github.com/deeplearning-wisc/gradnorm_ood). Then also put them in the `datasets` dir. The model BiT-S ResNetv2-101 is from [Big transfer](https://github.com/google-research/big_transfer).

## How to Run

#### Run GAIA-A on CIFAR10


```
python eval.py \
        -dataset cifar10 \
        -model_arch resnet \
        -model_name resnet34 \
        -num_classes 10 \
        -score GAIA \
        -cal_method cal_grad_value \
        -hook before_head \
        -data_dir ./datasets/ \
        -model_path ./checkpoint/models/cifar10_resnet34.pth \
        -batch_size 128 \
        -num_workers 4 \
        -cuda 0 \
```




#### Run GAIA-Z on CIFAR10

```
python eval.py \
        -dataset cifar10 \
        -model_arch resnet \
        -model_name resnet34 \
        -num_classes 10 \
        -score GAIA \
        -cal_method cal_zero \
        -hook bn \
        -data_dir ./datasets/ \
        -model_path ./checkpoint/models/cifar10_resnet34.pth \
        -batch_size 128 \
        -num_workers 4 \
        -cuda 0 \
```

#### Run GAIA on CIFAR100

```
python eval.py \
        -dataset cifar100 \
        -model_arch resnet \
        -model_name resnet34 \
        -num_classes 100 \
        -score GAIA \
        -cal_method [cal_grad_value / cal_zero] \
        -hook [before_head / bn] \
        -data_dir ./datasets/ \
        -model_path ./checkpoint/models/cifar100_resnet34.pth \
        -batch_size 128 \
        -num_workers 4 \
        -cuda 0 \
```

#### Run GAIA on ImageNet

```
python eval.py \
    -dataset imagenet \
    -model_arch resnetv2 \
    -model_name BiT-S-R101x1 \
    -num_classes 1000\
    -score GAIA \
    -cal_method [cal_grad_value / cal_zero] \
    -data_dir ./datasets/ \
    -hook before_head \
    -model_path [model_path] \
    -batch_size 16 \
    -num_workers 4 \
    -cuda 0 \
```

## References

This repo is based on [ODIN](https://github.com/facebookresearch/odin) and [GradNorm](https://github.com/deeplearning-wisc/gradnorm_ood).


## Citation

If you find our work useful, please cite as

```
@inproceedings{chen2023gaia,
  title={GAIA: Delving into Gradient-based Attribution Abnormality for Out-of-distribution Detection},
  author={Jinggang Chen, Junjie Li, Xiaoyang Qu, Jianzong Wang, Jiguang Wan, Jing Xiao},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


