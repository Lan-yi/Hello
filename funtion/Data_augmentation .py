import cv2
import numpy as np
from PIL import Image
import os
import cv2
#from matplotlib import pyplot as plt

project_path = 'D:/Study/ImageData/mohu/8/test/'
image_path = 'D:/Study/ImageData/origin/images'    #原始image路径
label_path = 'D:/Study/ImageData/origin/masks'    #原始mask路径

#############平移###############
# def Affine(img_path,output_path,imagename,affine_range):
#     img = cv2.imread(img_path)
#     operate = 'aff'
#     rows, cols,chanl = img.shape
#     M = np.float32([[1, 0, 0], [0, 1, affine_range]])
#     image1 = cv2.warpAffine(img, M, (cols, rows))
#     M = np.float32([[1, 0, 0], [0, 1, -1 * affine_range]])
#     image2 = cv2.warpAffine(img, M, (cols, rows))
#     M = np.float32([[1, 0, affine_range], [0, 1, 0]])
#     image3 = cv2.warpAffine(img, M, (cols, rows))
#     M = np.float32([[1, 0, -1 * affine_range], [0, 1, 0]])
#     image4 = cv2.warpAffine(img, M, (cols, rows))
#
#     Affine_range = str(affine_range)
#     output_imagepath = output_path + imagename + '_' + operate +'down' + Affine_range + '.bmp'
#     cv2.imwrite(output_imagepath,image1)
#     output_imagepath = output_path + imagename + '_' + operate + 'up' + Affine_range + '.bmp'
#     cv2.imwrite(output_imagepath,image2)
#     output_imagepath = output_path + imagename + '_' + operate + 'right' + Affine_range + '.bmp'
#     cv2.imwrite(output_imagepath,image3)
#     output_imagepath = output_path + imagename + '_' + operate + 'left' + Affine_range + '.bmp'
#     cv2.imwrite(output_imagepath,image4)
#
#
# filelist = os.listdir(image_path) #获取原始文件
# output_path = project_path + 'images/'    #输出目录
# for first_dir in filelist:
#     img_path = image_path + '/' + first_dir    #原始文件完整路径
#     imagename, suffix = os.path.splitext(first_dir)
#     affine_range = 0
#     for i in range(1,5):
#         affine_range = affine_range + 100
#         Affine(img_path, output_path, imagename,affine_range)
#
# filelist = os.listdir(label_path) #获取目录下的子文件
# output_path = project_path + 'masks/'
# for first_dir in filelist:
#     img_path = label_path + '/' + first_dir
#     imagename, suffix = os.path.splitext(first_dir)
#     affine_range = 0
#     for i in range(1,5):
#         affine_range = affine_range + 100
#         Affine(img_path, output_path, imagename,affine_range)
#
#
#
# ############镜像###############
# def mirror(img_path,output_path,imagename):
#     img = Image.open(img_path)
#     operate = 'mir'
#     img = img.transpose(0)
#     output_imagepath = output_path + imagename + '_' + operate +'.bmp'
#     img.save(output_imagepath)
#
# filelist = os.listdir(image_path) #获取主目录下的子文件
# output_path = project_path + 'images/'
# for first_dir in filelist:
#     img_path = image_path + '/' + first_dir
#     imagename, suffix = os.path.splitext(first_dir)
#     mirror(img_path, output_path, imagename)
#
# filelist = os.listdir(label_path) #获取主目录下的子文件
# output_path = project_path + 'masks/'
# for first_dir in filelist:
#     img_path = label_path + '/' + first_dir
#     imagename, suffix = os.path.splitext(first_dir)
#     mirror(img_path, output_path, imagename)


#############旋转###############
#输入文件地址，输出上级目录，角度，文件名
def rotateimage(img_path,output_path,angle,imagename):
    img = Image.open(img_path)
    operate = 'rot'
    Angle = str(angle)
    img = img.rotate(angle)
    # plt.imshow(img)
    # plt.show()
    output_imagepath = output_path + imagename + '_' + operate + Angle +'.bmp'
    img.save(output_imagepath)

filelist = os.listdir(image_path) #获取主目录下的子文件
output_path = project_path + 'images/'
for first_dir in filelist:
    angle = 0
    img_path = image_path + '/' + first_dir
    imagename, suffix = os.path.splitext(first_dir)
    for i in range(1,36):
        angle = angle + 10
        rotateimage(img_path, output_path, angle, imagename)

filelist = os.listdir(label_path) #获取主目录下的子文件
output_path = project_path + 'masks/'
for first_dir in filelist:
    angle = 0
    img_path = label_path + '/' + first_dir
    imagename, suffix = os.path.splitext(first_dir)
    for i in range(1,36):
        angle = angle + 10
        rotateimage(img_path, output_path, angle, imagename)