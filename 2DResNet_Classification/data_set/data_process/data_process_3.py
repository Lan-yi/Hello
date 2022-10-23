import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import cv2
import os
import shutil
from PIL import Image

#---------------------#
#改病号，日期，分级；
#主文件夹和N
#--------------------#

# patient = '2747472'    #病人编号
# date = '20191219'     #T3日期

patient = '2693114'    #病人编号
date = '20190828'     #T3日期

#path_DCM = 'D:/Pami/NAC_Data/GE/' + patient + '/' + date + '/'   #主文件夹
path_DCM = 'D:/Pami/NAC_Data/PHI/' + patient + '/' + date + '/'
###具体文件###
label_path = path_DCM + 'lesion/lesion.nii.gz'    #病灶标记文件
N = 'N6'
folder_N5N6 = path_DCM + N + '/'                    #次文件夹 GE-N5,PHI-N6
level = 2    #分级
index = 35    #PCR无标记文件另取索引
label_BST_N = path_DCM + 'BST/BST_N.nii.gz'
label_BST_P = path_DCM + 'BST/BST_P.nii.gz'

#####################################
#输入对应label找最大病灶层，返回最大层层数
def getmaxsliceindex(Mask):
    mask = sitk.ReadImage(Mask)
    maskArray = sitk.GetArrayFromImage(mask)
    maskArray = maskArray.astype(np.int16)
    x,y,z = mask.GetSize()
    print(x,y,z)
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
def getmaxslice_ROI(label_BST_N,label_BST_P,max,dcm2D):
    Max = str(max)
    if level == 5 :
        folder = 'PCR'
    else: folder = 'nPCR'
    print(folder)
    output_dcm = 'D:/Study/Dataset/PCR_nPCR/' + folder + '/' + patient + '_T3_' + N + '_' + Max + '.png'    #输出地址

    bstn = sitk.ReadImage(label_BST_N)
    bstnarray = sitk.GetArrayFromImage(bstn)
    bstnarray = bstnarray[max, :, :]

    bstp = sitk.ReadImage(label_BST_P)
    bstparray = sitk.GetArrayFromImage(bstp)
    bstparray = bstparray[max, :, :]

    dcm = sitk.ReadImage(dcm2D)
    dcmarray = sitk.GetArrayFromImage(dcm)
    z,y,x = dcmarray.shape

    bstnarray = bstnarray + bstparray
    dcmarray[0, :, :]  = dcmarray[0, :, :] * bstnarray
    dcmarray = dcmarray.astype('uint16')

    high = np.max(dcmarray)
    low = np.min(dcmarray)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (dcmarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
    dcmarray = (newimg * 1000).astype('uint16')
    cv2.imwrite(output_dcm, dcmarray[0,:,:])

    #灰度图转3通道
    img = Image.open(output_dcm).convert('RGB')
    imgarray = np.array(img)
    imgarray = imgarray.astype('uint16')
    imgarray[:, :, 0] = dcmarray
    imgarray[:, :, 1] = dcmarray
    imgarray[:, :, 2] = dcmarray
    cv2.imwrite(output_dcm, imgarray)
    print(len(bstnarray))
    print(len(bstnarray[0]))
    print(output_dcm)
    return output_dcm


##################################
def getmax_dcm2D(folder_N5N6,max):
    files = os.listdir(folder_N5N6)
    files.sort()    #按顺序读文件
    max = max + 1
    filename = folder_N5N6 + files[max]    #dcm2D文件名
    #Max = str(max)
    #output_path =  'D:/Pami/Dataset/PCR_nPCR/' + patient + '_T3_N6_' + Max + '.dcm'
    #output_path =  'D:/Pami/Dataset/PCR_nPCR'
    #file_dcm = output_path + '/' + files[max]
    #shutil.copy(filename,output_path)
    return filename



#################################
max = getmaxsliceindex(label_path)                     #最大切片索引
print(getmaxsliceindex(label_path))
if max==0:
    max = index
print(max)
filename = getmaxslice_ROI(label_BST_N,label_BST_P,max,getmax_dcm2D(folder_N5N6,max))
getmaxslice_ROI(label_BST_N,label_BST_P,max+1,getmax_dcm2D(folder_N5N6,max))
getmaxslice_ROI(label_BST_N,label_BST_P,max-1,getmax_dcm2D(folder_N5N6,max))


#################################
img = cv2.imread(filename,-1)
imgarray = np.array(img)

high = np.max(imgarray)
low = np.min(imgarray)
lungwin = np.array([low * 1., high * 1.])
newimg = (imgarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
imgarray = (newimg * 255).astype('uint8')  # GE将像素值扩展到[0,3000]，PHI[0.5500]

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", imgarray)
cv2.waitKey(0)
cv2.destroyAllWindows()

