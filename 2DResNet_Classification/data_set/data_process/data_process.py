import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
import SimpleITK as sitk
import cv2
import tifffile as tiff

patient = '192297'
date = '20190312'
path_DCM = 'D:/Pami/Dataset/GE/' + patient + '/' + date + '/'
label_path1 = path_DCM+'lesion/lesion.nii.gz'
file_dcm = path_DCM+'N5/N5.nii.gz'


############################
#输入对应label找最大病灶层
def getmaxslice(Mask):
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


############################
#统一spacing的函数，输入图像
def resample_image(itk_image, out_spacing=[1.0, 1.0, 2.0]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    # print('size:', original_size)
    # print('spacing:', original_spacing)

    # 根据输出out_spacing设置新的size
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkBSpline)
    # print('newsize:', out_size)
    # print('newspacing:', out_spacing)

    return resample.Execute(itk_image)


############################
def convertpng(path_DCM,label_path1,file_dcm,max):
    #参考exp5，dicom序列中读入一层转为二维
    dcm_3D = sitk.ReadImage(file_dcm)
    resample_image(dcm_3D)

    dcm_array = sitk.GetArrayFromImage(dcm_3D)
    origin = list(dcm_3D.GetOrigin())
    spacing = list(dcm_3D.GetSpacing())

    dcm_array = dcm_array[max,:,:]                     #取某一切片对应数组部分
    dcm_array = dcm_array.astype(np.ushort)
    dcm_2D = sitk.GetImageFromArray(dcm_array)
    dcm_2D.SetOrigin(origin[:2])
    dcm_2D.SetSpacing(spacing[:2])


    #灰度归一化
    # resacleFilter = sitk.RescaleIntensityImageFilter()
    # resacleFilter.SetOutputMaximum(255)
    # resacleFilter.SetOutputMinimum(0)
    # dcm_2D = resacleFilter.Execute(dcm_2D)

    high = np.max(dcm_array)
    low = np.min(dcm_array)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (dcm_array - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
    newimg = (newimg * 3000).astype('uint16')  # 将像素值扩展
    #src = sitk.GetImageFromArray(newimg)
    #src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    #newimg = sitk.GetArrayFromImage(src_RGB)
    #newimg = cv2.cvtColor(newimg, cv2.COLOR_GRAY2BGR)

    np.repeat(mask[..., np.newaxis], 3, 2)
    Max = str(max)
    output_png_path =  'D:/Pami/Dataset/fenlei/' + patient + '_T3_N6_' + Max + '.tiff'
    #cv2.imwrite(output_png_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(output_png_path, newimg,  (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
    #tiff.imwrite(output_png_path,newimg)





#################################
max = getmaxslice(label_path1)                     #最大切片索引
print(getmaxslice(label_path1))

#max = 30
convertpng(path_DCM,label_path1,file_dcm,max)
convertpng(path_DCM,label_path1,file_dcm,max+1)
convertpng(path_DCM,label_path1,file_dcm,max-1)




#image = cv2.imread('D:/Pami/Dataset/fenlei/192297_T3_N6_26.png')
#MAX = str(max)

#sitk.WriteImage(dcm_2D, 'D:/Pami/Dataset/fenlei/2928924_guiyihua_N6.png')
#sitk.WriteImage(dcm_2D, 'D:/Pami/Dataset/fenlei/2928924.png')    #切片保存到此处
#2928924是384*384的



