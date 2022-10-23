import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
import SimpleITK as sitk
import cv2
import os
import shutil
import time

patient = '192297'
date = '20190312'
path_DCM = 'D:/Pami/Dataset/GE/' + patient + '/' + date + '/'
label_path1 = path_DCM+'lesion/lesion.nii.gz'
file_dcm_dir = path_DCM+'N5/'
label_BST_N = path_DCM + 'BST/BST_N.nii.gz'
label_BST_P = path_DCM + 'BST/BST_P.nii.gz'

############################
#输入对应label找最大病灶层
def getmaxsliceindex(Mask):
    mask = sitk.ReadImage(Mask)
    maskArray = sitk.GetArrayFromImage(mask)
    maskArray = maskArray.astype(np.int16)
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



################################
#输入BST标注和dcm2D文件框选ROI
def getmaxslice_BSTlabel(label_BST_N,label_BST_P,max,output_dcm):
    bstn = sitk.ReadImage(label_BST_N)
    bstnarray = sitk.GetArrayFromImage(bstn)
    bstnarray = bstnarray[max, :, :]

    bstp = sitk.ReadImage(label_BST_P)
    bstparray = sitk.GetArrayFromImage(bstp)
    bstparray = bstparray[max, :, :]

    dcm = sitk.ReadImage(output_dcm)
    dcmarray = sitk.GetArrayFromImage(dcm)

    # for i in range(0, len(bstnarray)):
    #     for j in range(0, len(bstnarray[0])):
    #         bstnarray[i][j] = bstnarray[i][j] + bstparray[i][j]
    #         if bstnarray[i][j] == 0:
    #             dcmarray[0][i][j] = 0

    bstnarray = bstnarray + bstparray
    dcmarray[0, :, :]  = dcmarray[0, :, :] * bstnarray
    dcmarray.dtype = 'uint16'

    high = np.max(dcmarray)
    low = np.min(dcmarray)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (dcmarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
    dcmarray = (newimg * 3000).astype('uint16')  # GE将像素值扩展到[0,3000]，PHI[0.5500]
    #
    # image = np.expand_dims(dcmarray, axis=2)
    # dcmarray = np.concatenate((image, image, image), axis=-1)
    #
    #dcm = cv2.cvtColor(dcmarray, cv2.COLOR_GRAY2BGR)
    print(len(bstnarray))

    #dcm = sitk.GetImageFromArray(dcmarray)
    #sitk.WriteImage(dcm,output_dcm)
    cv2.imwrite(output_dcm, dcmarray, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])


    print(len(bstnarray))
    print(len(bstnarray[0]))




##################################
def getmaxslice(file_dcm_dir,max):
    files = os.listdir(file_dcm_dir)
    files.sort()    #按顺序读文件
    max = max + 1
    filename = file_dcm_dir + files[max]
    Max = str(max)
    #output_path =  'D:/Pami/Dataset/PCR_nPCR/' + patient + '_T3_N6_' + Max + '.dcm'
    output_path =  'D:/Pami/Dataset/PCR_nPCR'
    file_dcm = output_path + '/' + files[max]
    shutil.copy(filename,output_path)
    return file_dcm



#################################
max = getmaxsliceindex(label_path1)                     #最大切片索引
print(getmaxsliceindex(label_path1))
#max = 30
print(getmaxslice(file_dcm_dir,max))
#output_dcm = 'D:/Pami/Dataset/PCR_nPCR/IM00238.dcm'

time_start = time.time()
getmaxslice_BSTlabel(label_BST_N,label_BST_P,max,getmaxslice(file_dcm_dir,max))
time_end = time.time()
print('点乘操作 totally cost',time_end-time_start)







#image = cv2.imread('D:/Pami/Dataset/fenlei/192297_T3_N6_26.png')


#sitk.WriteImage(dcm_2D, 'D:/Pami/Dataset/fenlei/2928924.png')    #切片保存到此处




#     ############################
# def convertpng(path_DCM,label_path1,file_dcm,max):
#     #参考exp5，dicom序列中读入一层转为二维
#     dcm_3D = sitk.ReadImage(file_dcm)
#     #resample_image(dcm_3D)
#     dcm_array = sitk.GetArrayFromImage(dcm_3D)
#     origin = list(dcm_3D.GetOrigin())
#     spacing = list(dcm_3D.GetSpacing())
#     dcm_array = dcm_array[max,:,:]                     #取某一切片对应数组部分
#     dcm_array = dcm_array.astype(np.ushort)
#     dcm_2D = sitk.GetImageFromArray(dcm_array)
#     dcm_2D.SetOrigin(origin[:2])
#     dcm_2D.SetSpacing(spacing[:2])
#
#     Max = str(max)
#     output_png_path =  'D:/Pami/Dataset/fenlei/' + patient + '_T3_N6_' + Max + '.png'
#     cv2.imwrite(output_png_path, dcm_2D, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
#
