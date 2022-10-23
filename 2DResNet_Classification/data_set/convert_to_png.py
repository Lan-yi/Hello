import numpy as np
import SimpleITK as sitk
import cv2

path_DCM = 'D:/Pami/lesion_mark/NAC_Breast_data_jyl/PHI/2928924/20201021/'



def convert_from_dicom_to_jpg(img, low_window, high_window, save_path):
    lungwin = np.array([low_window * 1., high_window * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  # 归一化
    newimg = (newimg * 255).astype('uint8')  # 将像素值扩展到[0,255]
    cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


#if __name__ == '__main__':
# 下面是将对应的dicom格式的图片转成jpg
dcm_image_path = path_DCM+'N6.nii.gz' # 读取dicom文件
output_png_path: str = 'D:/Pami/Dataset/fenlei/2928924_convert_N6.png'
ds_array = sitk.ReadImage(dcm_image_path)  # 读取dicom文件的相关信息
img_array = sitk.GetArrayFromImage(ds_array)  # 获取array
# SimpleITK读取的图像数据的坐标顺序为zyx，即从多少张切片到单张切片的宽和高，此处我们读取单张，因此img_array的shape
# 类似于 （1，height，width）的形式
shape = img_array.shape
img_array = np.reshape(img_array, (shape[1], shape[2]))  # 获取array中的height和width
high = np.max(img_array)
low = np.min(img_array)

convert_from_dicom_to_jpg(img_array, low, high, output_png_path)  # 调用函数，转换成jpg文件并保存到对应的路径

