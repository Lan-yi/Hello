import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import natsort
import shutil
from PIL import Image
import nibabel as nib

#---------------------#
#txt转为list,提取特定文件
#--------------------#
patient = []
date = []
catalog = []
level = []
index = []
txtname = 'C:/Users/86156/Desktop/txt/make_segdataset_ALL.txt'
with open(txtname) as file_object:
    lines = file_object.readlines()
    for line in lines:
        patient.append(line.split()[0])  # 病人编号
        date.append(line.split()[1])     # T3日期
        catalog.append(line.split()[2])
        level.append(line.split()[3])    #分级
        if len(line.split()) == 5:
            index.append(line.split()[4])
        else: index.append(0)

def addlabel(labelpath1,labelpath2,output_filepath):
    label1 = nib.load(labelpath1)
    label2 = nib.load(labelpath2)
    # 将仿射矩阵和头文件保存下来
    affine = label1.affine
    hdr = label1.header
    data1 = label1.get_data()
    data2 = label2.get_data()
    new = data1 | data2
    print(np.min(new), np.max(new))

    # 形成新的nii文件
    new_nii = nib.Nifti1Image(new, affine, hdr)
    nib.save(new_nii, output_filepath)


#######catch file

for i in range(len(lines)):
    dir_DCM = 'D:/Pami/NAC_Data/' + catalog[i] + '/' + patient[i] + '/' + date[i] + '/'  # 目录
    if catalog[i] == 'GE':
        N = 'N5'
    elif catalog[i] == 'PHI':
        N = 'N6'
    else:
        print("catalog error")
        break
    folder_N5N6 = dir_DCM + N + '/'  # 次文件夹 GE-N5,PHI-N6
    image_path =  folder_N5N6 + N + '.nii.gz'
    label_path_N = dir_DCM + 'BST/BST_N.nii.gz'  # 标记文件
    label_path_P = dir_DCM + 'BST/BST_P.nii.gz'  # 标记文件
    dir_image = 'D:/Study/Dataset/NACData_BST_ALL/img_lesion/images/' + patient[i] + '_' + N + '.nii.gz'
    output_filepath = 'D:/Study/Dataset/NACData_BST_ALL/img_lesion/labels/' + patient[i] + '_' + N + '.nii.gz'
    shutil.copyfile(image_path, dir_image)
    addlabel(label_path_N, label_path_P, output_filepath)
    #shutil.copyfile(label_path_N, dir_label)
    print(patient[i])


