from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from networks import methods
from data import datasets
import numpy as np
import pandas as pd
import time

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, help='Optional ID dataset: cifar10 | cifar100 | imagenet', default='cifar10')
parser.add_argument('-model_arch', type=str, help='Optional model: resnet | resnetv2 | wresnet | wresnet', default='resnet')
parser.add_argument('-model_name', type=str, help='Optional model: resnet34 | BiT-S-R101x1 | wrn_40_2', default='resnet34')
parser.add_argument('-cal_method', type=str, 
                    help='Optional method: cal_zero|cal_grad_value', 
                    default='cal_zero')
parser.add_argument('-hook', type=str, help='hook type', default='bn')
parser.add_argument('-score', type=str, help='score method', default='GAIA')
parser.add_argument('-data_dir', type=str, help='Data load path', default='/home/ljj/data')
parser.add_argument('-model_path', type=str, help='Model load path', default='./checkpoint/models/cifar10_resnet34.pth')
parser.add_argument('-save_dir', type=str, help='Data save path', default='./checkpoint/records/')

parser.add_argument('-batch_size', type=int, help='Batch size', default=64)
parser.add_argument('-num_workers', type=int, help='Num_workers', default=4)
parser.add_argument('-cuda', type=int, help='cuda use', default='0')
parser.add_argument('-num_classes', type=int, help='number of classes', default=10)


args = parser.parse_args()
torch.cuda.set_device(args.cuda)
print(args)

evaluator = methods.Methods(args)
in_test, ood_datasets, ood_name = datasets.get_datasets(args)
evaluator.get_scores(in_test, ood_name, ood_datasets)