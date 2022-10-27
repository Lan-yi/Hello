import os
import sys
import numpy as np
import math
import random

import torch
# print(torch.__version__)
from torch.utils.data import DataLoader

import nibabel as nb
import SimpleITK as sitk
import multiprocessing
import matplotlib.pyplot as plt

from model import VNet

from file_load import file_read
from dataloader_test import dataset
# from test import test
from evaluate import test


device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model = VNet(n_channels=1, n_classes=3, n_filters=16, normalization='groupnorm', activation = 'ReLU')
model = model.to(device)
model.load_state_dict(torch.load(r'.\model\vnet_breast_0925.pth', map_location=device))
dir_image = 'D:/Study/Dataset/NACData_VNet/img_lesion/images/'
path_save = 'D:/Study/Dataset/NACData_VNet/FGT_SEG_N0/'
#dir_image = r'D:\Study\Dataset\NACData_VNet\img_lesion\images'
txtname = 'C:/Users/86156/Desktop/txt/path_N5N6.txt'
with open(txtname) as file_object:
    lines = file_object.readlines()
    for line in lines:
        filename = line.split('.')[0]
        niifile = file_read(os.path.join(dir_image, line.split()[0]))
        test_loader = dataset(niifile, mode='test')
        new, prob_bst, prob_fgt = test(model, test_loader, path_save, device, filename)






# #pathlist1 = r'F:\NAC_Breast_data_jyl\path_no_label.txt'
# filepaths1 = [files.strip().split() for files in open(pathlist1)]
# # filepaths2 = [files.strip().split() for files in open(pathlist2)]
# # filepaths = filepaths1 + filepaths2
# for file_i in filepaths1:
#     filepath = file_i[0]
#     niifile = file_read(os.path.join(filepath, r'N0\N0.nii.gz'))
#     test_loader = dataset(niifile, mode = 'test')
#     path_save = os.path.join(filepath, 'FGT_SEG_N0')
#     if not os.path.isdir(path_save):
#         os.makedirs(path_save)
#     new, prob_bst, prob_fgt = test(model,test_loader, path_save, device)
#
#
#
# # one test
# file_path = r'D:\Pami\NAC_Data\PHI\2898346\20200902'
# niifile = file_read(os.path.join(file_path, r'N0\N0.nii.gz'))
#
# test_loader = dataset(niifile, mode = 'test')
# print(len(test_loader))
# path_save = os.path.join(file_path, 'FGT_SEG_N0')
# if not os.path.isdir(path_save):
#     os.makedirs(path_save)
# new, prob_bst, prob_fgt = test(model, test_loader, path_save, device)



# one test
# file_path = r'F:\NAC_Breast_data_jyl\PHI\2898346\20200902'
# niifile = file_read(os.path.join(file_path, r'N0\N0.nii.gz'))
# test_loader = dataset(niifile, mode = 'test')
# print(len(test_loader))
# path_save = os.path.join(file_path, 'FGT_SEG_N0')
# if not os.path.isdir(path_save):
#     os.makedirs(path_save)
# new, prob_bst, prob_fgt = test(model, test_loader, path_save, device)