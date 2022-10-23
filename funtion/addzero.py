import cv2
from PIL import Image
import os
import numpy as np

#D:/Pami/Dataset/jinghang/train/img_' + str(i) + '.jpg'
dir = 'D:/Study/ImageData/mohu/'
image_dir = 'D:/Study/ImageData/mohu/image/origin/'
label_dir = 'D:/Study/ImageData/mohu/label/origin/'

def addzero(file_path):
    inputImage = cv2.imread(file_path, 1)
    modelImgpath = 'D:/Study/ImageData/Im3.2ModelInv.bmp'
    modelimg = Image.open(modelImgpath)
    modelimgSize = modelimg.size
    img = Image.open(file_path)
    imgSize = img.size  # 大小/尺寸
    w = img.width  # 图片的宽
    h = img.height

    left = int((modelimgSize[0] - w) / 2)
    right = left

    if w + left + right != modelimgSize[0]:
        left = modelimgSize[0] - w - right

    top = int((modelimgSize[1] - h) / 2)
    bottom = top
    if h + top + bottom != modelimgSize[1]:
        top = modelimgSize[1] - h - bottom

    outputImage = cv2.copyMakeBorder(inputImage, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # 输出的图片文件夹
    #'D:/Pami/Dataset/jinghang/Im3.6ModelInv.bmp'

    cv2.imwrite(file_path, outputImage)

image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)
image_files.sort()    #按顺序读文件
label_files.sort()

for i in range(0, 36):
    image_filename = image_dir + image_files[i]
    label_filename = label_dir + label_files[i]

    print('image:', image_filename)
    print('label:', label_filename)
    addzero(image_filename)
    addzero(label_filename)

# 如果将图片补全为1280*720，输入的图片不能大于这个尺寸，高和宽可以自己修改
# for i in range(1, 1099):
#     # 输入的图片文件夹
#     inputImage = cv2.imread('D:/Pami/Dataset/jinghang/train/img_' + str(i) + '.jpg', 1)
#
#     file_path = 'rotate2/img_' + str(i) + '.jpg'
#     img = Image.open(file_path)
#     imgSize = img.size  # 大小/尺寸
#     w = img.width  # 图片的宽
#     h = img.height
#
#     left = int((1280 - w) / 2)
#     right = left
#     if w + left + right != 1280:
#         left = 1280 - w - right
#
#     top = int((720 - h) / 2)
#     bottom = top
#     if h + top + bottom != 720:
#         top = 720 - h - bottom
#
#     outputImage = cv2.copyMakeBorder(inputImage, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
#     # 输出的图片文件夹
#     cv2.imwrite('rotate3/img_' + str(i) + '.jpg', outputImage)
