import os
import torch
from vnet.train.model3 import VNet
from file_load import file_read
from dataloader_train import dataset_1121
from evaluate_train import train
from torch.utils.data import DataLoader
import torch.optim as optim
import atexit
from torch import nn
import nibabel as nb
import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from nilearn.image import resample_img
from scipy import ndimage
from typing import Dict, Sequence, Optional, Callable

from pathlib import Path
#
def save_nii(imarr, affine, pth, name):
    nii_img = nb.Nifti1Image(imarr, affine=affine)
    # nib.save(nii_img, pth + name)
    nb.save(nii_img, os.path.join(pth, name))

dir_image = r'../dataset/NACData_VNet_hundreds'    #根目录
txtname = '../dataset/NACData_VNet_hundreds/train.txt'
save_pth = r'../dataset/NACData_VNet_hundreds/maskN5N6'
with open(txtname) as file_object:
    lines = file_object.readlines()
    lines = [line.strip() for line in lines]
    imglist = lines

for i in range(len(imglist)):
    ith_info = imglist[i].split(" ")
    filename = ith_info[0].split('.')[0].split('\\')[1] + '.nii.gz'

    img_name = os.path.join(dir_image, ith_info[0])
    turelabel_name = os.path.join(dir_image, ith_info[1])
    prelabel_name = os.path.join(dir_image, ith_info[2])
    if not os.path.isfile(img_name):
        print(img_name)

    assert os.path.isfile(img_name)
    assert os.path.isfile(turelabel_name)
    assert os.path.isfile(prelabel_name)

    # 读为数组
    volume_nifty = nb.load(img_name)
    mask_nifty = nb.load(prelabel_name)

    volume_orig_affine = volume_nifty.affine
    volume_orig_shape = volume_nifty.get_fdata().shape

    volume = volume_nifty.get_fdata()
    mask = mask_nifty.get_fdata()
    mask[np.where(mask == 2)] = 1
    mask[np.where(mask == 3)] = 1

    max = np.max(mask)


    volume = volume * mask

    save_nii(volume, volume_orig_affine, save_pth, filename)








