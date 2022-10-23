import imageio
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from skimage.transform import resize
import SimpleITK as sitk
import cv2
import os
import shutil
import time
from PIL import Image

############################
#输入对应label找最大病灶层
def getmaxsliceindex(Mask):
    mask = sitk.ReadImage(Mask)
    maskArray = sitk.GetArrayFromImage(mask)
    maskArray = maskArray.astype(np.int16)
    x,y,z = mask.GetSize()
    # maxvol = 0
    # maxslice = 0
    countlist = []
    for i in range(0,z):
        countlist.append(np.sum(maskArray[i] == 1))
    return np.argmax(np.array(countlist))

################################
#输入BST标注和dcm2D文件框选ROI
def getmaxslice_BSTlabel(label_BST_N,label_BST_P,max,file_dcm):
    bstn = sitk.ReadImage(label_BST_N)
    bstnarray = sitk.GetArrayFromImage(bstn)
    bstnarray = bstnarray[max, :, :]

    bstp = sitk.ReadImage(label_BST_P)
    bstparray = sitk.GetArrayFromImage(bstp)
    bstparray = bstparray[max, :, :]

    dcm = sitk.ReadImage(file_dcm)
    dcmarray = sitk.GetArrayFromImage(dcm)
    z,y,x = dcmarray.shape


    bstnarray = bstnarray + bstparray
    dcmarray[0, :, :]  = dcmarray[0, :, :] * bstnarray
    dcmarray = dcmarray.astype('float64')

    high = np.max(dcmarray)
    low = np.min(dcmarray)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (dcmarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
    dcmarray = (newimg * 1000).astype('float64')  # GE将像素值扩展到[0,3000]，PHI[0.5500]

    #dcm = cv2.cvtColor(dcmarray, cv2.COLOR_GRAY2BGR)
    #dcm = sitk.GetImageFromArray(dcmarray)
    #sitk.WriteImage(dcm,output_dcm)
    #
    # dcmarray_3 = np.zeros((x, y, 3), dtype=float)
    # dcmarray_3 = dcmarray_3.astype(np.float64)
    # dcmarray_3[:,:,0] = dcmarray[0,:,:]
    # dcmarray_3[:,:,1] = dcmarray[0,:,:]
    # dcmarray_3[:,:,2] = dcmarray[0,:,:]
    #a = np.transpose(dcmarray_3,(2,0,1))
    #Image.fromarray(dcmarray_3).save(output_dcm)
    #cv2.imwrite(output_dcm, dcmarray[0,:,:], [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
    temp = '\\'.join(file_dcm.split('/')[:-1])
    output_dcm = os.path.join(temp,'test.tif')
    cv2.imwrite(output_dcm, dcmarray[0])

    print(len(bstnarray))
    print(len(bstnarray[0]))

def getmaxslice(file_dcm_dir,max):

    # Max = str(max)files = os.listdir(file_dcm_dir)
    #     files.sort()    #按顺序读文件
    #     # print(files)
    #     max = max + 1
    #     filename = file_dcm_dir + files[max]
    #output_path =  'D:/Pami/Dataset/PCR_nPCR/' + patient + '_T3_N6_' + Max + '.dcm'
    # output_path =  'D:/Pami/Dataset/PCR_nPCR'
    # file_dcm = output_path + '/' + files[max]
    # shutil.copy(filename,output_path)
    return filename

def getMaxLesionSlice(img,lesion,BST_N,BST_P,savepath):
    maxSliceindex = getmaxsliceindex(lesion)
    # print((maxSliceindex))
    imgArr = sitk.GetArrayFromImage(sitk.ReadImage(img))

    imgArr = (imgArr/3000) * 4000
    imgArr[imgArr > 5500] = 5500
    imgArr = imgArr.astype(np.int32)
    # print(np.max(imgArr))

    bstNArr = sitk.GetArrayFromImage(sitk.ReadImage(BST_N))
    bstPArr = sitk.GetArrayFromImage(sitk.ReadImage(BST_P))
    bstimgArr = imgArr * (bstNArr + bstPArr)
    # print(np.max(bstimgArr))
    cv2.imwrite(savepath, bstimgArr[maxSliceindex],[-1])
    # tifffile.imsave(savepath, bstimgArr[maxSliceindex])
    # sitk.WriteImage(bstimgArr[maxSliceindex], savepath)


#################################
patient = '192297'
date = '20190312'
path_DCM = 'D:/Pami/Dataset/GE/' + patient + '/' + date + '/'
lesion_path1 = path_DCM+'lesion/lesion.nii.gz'
img_path = path_DCM+'N5/N5.nii.gz'
BST_N = path_DCM + 'BST/BST_N.nii.gz'
BST_P = path_DCM + 'BST/BST_P.nii.gz'
max = getmaxsliceindex(lesion_path1)                     #最大切片索引
Max = str(max);
output_dcm =  'D:/Pami/Dataset/PCR_nPCR/' + patient + '_T3_N6_' + Max + '.tiff'
getMaxLesionSlice(img_path,lesion_path1,BST_N,BST_P,output_dcm)

print(np.max(cv2.imread(output_dcm,-1)))
plt.imshow(cv2.imread(output_dcm,-1))
plt.show()





