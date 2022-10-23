import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import cv2
import os
import natsort
import shutil
from PIL import Image
#---------------------#
#使用******diff******图像
#--------------------#
txtname = 'C:/Users/86156/Desktop/txt/make_dataset.txt'
with open(txtname) as file_object:
    lines = file_object.readlines()
#---------------------#
#改病号，日期，分级；
#主文件夹和N
#--------------------#
# i= 28
# patient = lines[i].split()[0]    #病人编号
# date = lines[i].split()[1]  # T3日期
# catalog = lines[i].split()[2]
# level = int(lines[i].split()[3])   #分级

patient = '3139362'    #病人编号
date = '20220118'  # T3日期
catalog = 'PHI'
level = 5    #分级
index = 45  #PCR无病灶标记文件另取索引

def set_parameter(patient,date,catalog):
    global path_DCM
    global label_path
    global folder_N5N6
    global label_BST_N
    global label_BST_P
    global N
    path_DCM = 'D:/Pami/NAC_Data/' + catalog +'/' + patient + '/' + date + '/'
    ###具体文件###
    label_path = path_DCM + 'lesion/lesion.nii.gz'    #病灶标记文件
    if catalog == 'GE':
        N = 'N5'
    elif catalog == 'PHI':
        N = 'N6'
    else:print("catalog error")
    folder_N5N6 = path_DCM + N + '/'                    #次文件夹 GE-N5,PHI-N6
    label_BST_N = path_DCM + 'BST/BST_N.nii.gz'
    label_BST_P = path_DCM + 'BST/BST_P.nii.gz'


#####################################
#输入对应label找最大病灶层，返回最大层层数
def getmaxsliceindex(Mask):
    mask = sitk.ReadImage(Mask)
    maskArray = sitk.GetArrayFromImage(mask)
    maskArray = maskArray.astype(np.int16)
    x,y,z = mask.GetSize()
    print("shape:" , x,y,z)
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
#输入BST标注和diff图像框选ROI
def getmaxslice_ROI(label_BST_N,label_BST_P,max,diff):
    Max = str(max)
    if level == 5 :
        folder = 'PCR'
    else: folder = 'nPCR'
    print(folder)
    output_dcm = 'D:/Study/Dataset/PCR_nPCR/' + folder + '/' + patient + '_T3_diff_' + Max + '.png'    #输出地址

    bstn = sitk.ReadImage(label_BST_N)
    bstnarray = sitk.GetArrayFromImage(bstn)
    bstnarray = bstnarray[max, :, :]

    bstp = sitk.ReadImage(label_BST_P)
    bstparray = sitk.GetArrayFromImage(bstp)
    bstparray = bstparray[max, :, :]

    dcm = sitk.ReadImage(diff)
    dcmarray = sitk.GetArrayFromImage(dcm)
    z,y,x = dcmarray.shape

    bstnarray = bstnarray + bstparray
    outarray  = dcmarray[max, :, :] * bstnarray
    #outarray = outarray.astype('uint16')

    if catalog == 'GE':
        high = 3000
    else:
        high = 5500                         #GE3000 PHI5500
    #maxvalue = np.max(dcmarray)

    newimg = outarray / high    # 归一化
    outarray = (newimg * 1000).astype('int16')
    # for i in range(0,x-1):
    #     for j in range(0,y-1):
    #         if dcmarray[0,i,j] > 1000:
    #             dcmarray[0, i, j] = 1000

    # im = Image.fromarray(outarray)
    # im.save(output_dcm)
    cv2.imwrite(output_dcm, outarray)

    #灰度图转3通道
    img = Image.open(output_dcm).convert('RGB')
    imgarray = np.array(img)
    imgarray = imgarray.astype('int16')
    imgarray[:, :, 0] = outarray
    imgarray[:, :, 1] = outarray
    imgarray[:, :, 2] = outarray
    cv2.imwrite(output_dcm, imgarray)
    # print(len(bstnarray))
    # print(len(bstnarray[0]))
    print(output_dcm)
    return output_dcm


#################################
set_parameter(patient,date,catalog)
max = getmaxsliceindex(label_path)                     #最大切片索引
diff = path_DCM + 'diff/diff.nii.gz'
print("初始最大层索引：" , max)
if max==0:
    max = index
print("最终最大层索引：" , max)
print("---------------")
filename = getmaxslice_ROI(label_BST_N,label_BST_P,max,diff)
print("---------------")
getmaxslice_ROI(label_BST_N,label_BST_P,max+1,diff)
print("---------------")
getmaxslice_ROI(label_BST_N,label_BST_P,max-1,diff)


##############查看图片###################
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