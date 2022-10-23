import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import cv2
import os

patient = '995701'    #病人编号
date = '20210830'  # T3日期
path_DCM = 'D:/Pami/NAC_Data/PHI/' + patient + '/' + date + '/'   #主文件夹
###具体文件###
label_path = path_DCM + 'lesion/lesion.nii.gz'    #病灶标记文件
def getmaxsliceindex(Mask):
    mask = sitk.ReadImage(Mask)
    maskArray = sitk.GetArrayFromImage(mask)
    #maskArray = np.array(maskArray, dtype='uint8')
    #maskArray = maskArray.astype('int')
    x,y,z = mask.GetSize()
    maxvol = 0
    maxslice = 0
    volArray = np.zeros(z)
    for i in range(0,z):
        volArray[i] = np.sum(maskArray[i] == 1)
    for i in range(0, z):
        if volArray[i] > maxvol:
            maxvol = volArray[i]
            maxslice = i
    return maxslice



print(getmaxsliceindex(label_path))