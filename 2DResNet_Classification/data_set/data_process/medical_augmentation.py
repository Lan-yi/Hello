import numpy as np
import os
import cv2
#from matplotlib import pyplot as plt


dir = '/train/PCR'
image_path = 'D:/Study/Dataset/medicaldata_uint16_png_3_2_1' + dir
outputpath = 'D:/Study/Dataset/medicaldata_uint16_png_3_2_3' + dir
outputpath = outputpath + '/'


######旋转########
# def rotateimage(img_path,outputpath,filename):
#     img = cv2.imread(img_path,-1)
#     #img180 = np.rot90(img, 2)
#     img180 = cv2.rotate(img, cv2.ROTATE_180)
#     # cv2.imshow("rotate", img180)
#     # cv2.waitKey(0)
#     output_imagepath = outputpath + 'rotate' + filename
#     cv2.imwrite(output_imagepath, img180)
#
# filelist = os.listdir(image_path) #获取主目录下的子文件
# for first_dir in filelist:
#     img_path = image_path + '/' + first_dir
#     #imagename, suffix = os.path.splitext(first_dir)
#     rotateimage(img_path, outputpath, first_dir)

############平移###############
def Affine(img_path,output_path,filename,affine_range):
    img = cv2.imread(img_path,-1)
    operate = 'aff'
    rows, cols, chanl = img.shape
    M = np.float32([[1, 0, 0], [0, 1, affine_range]])
    image1 = cv2.warpAffine(img, M, (cols, rows))
    M = np.float32([[1, 0, 0], [0, 1, -1 * affine_range]])
    image2 = cv2.warpAffine(img, M, (cols, rows))
    M = np.float32([[1, 0, affine_range], [0, 1, 0]])
    image3 = cv2.warpAffine(img, M, (cols, rows))
    M = np.float32([[1, 0, -1 * affine_range], [0, 1, 0]])
    image4 = cv2.warpAffine(img, M, (cols, rows))

    Affine_range = str(affine_range)
    output_imagepath = output_path + imagename + '_' + operate + 'down' + Affine_range + '.png'
    cv2.imwrite(output_imagepath,image1)
    output_imagepath = output_path + imagename + '_' + operate + 'up' + Affine_range + '.png'
    cv2.imwrite(output_imagepath,image2)
    # output_imagepath = output_path + imagename + '_' + operate + 'right' + Affine_range + '.png'
    # cv2.imwrite(output_imagepath,image3)
    # output_imagepath = output_path + imagename + '_' + operate + 'left' + Affine_range + '.png'
    # cv2.imwrite(output_imagepath,image4)


filelist = os.listdir(image_path) #获取原始文件

for first_dir in filelist:
    img_path = image_path + '/' + first_dir    #原始文件完整路径
    imagename, suffix = os.path.splitext(first_dir)
    affine_range = 0
    for i in range(1,5):
        affine_range = affine_range + 10
        Affine(img_path, outputpath, imagename,affine_range)



############镜像###############
def mirror(img_path,output_path,imagename):
    img = cv2.imread(img_path,-1)
    operate = 'mir'
    imgflip = cv2.flip(img,1)
    output_imagepath = output_path + imagename + '_' + operate +'.png'
    cv2.imwrite(output_imagepath, imgflip)

filelist = os.listdir(image_path) #获取主目录下的子文件

for first_dir in filelist:
    img_path = image_path + '/' + first_dir
    imagename, suffix = os.path.splitext(first_dir)
    mirror(img_path, outputpath, imagename)



