import cv2
import numpy as np
from PIL import Image
import os
# from matplotlib import pyplot as plt


dir = '/test/nPCR'
image_path = 'D:/Study/Dataset/medicaldata_uint16_png_3_2_1' + dir
outputpath = 'D:/Study/Dataset/medicaldata_uint16_png_3_2_2' + dir
outputpath = outputpath + '/'
#outputpath = 'D:/Study/Dataset/PCR_nPCR/output_GE/'

def rotateimage(img_path,outputpath,filename):
    img = cv2.imread(img_path,-1)
    #img180 = np.rot90(img, 2)
    img180 = cv2.rotate(img, cv2.ROTATE_180)
    # cv2.imshow("rotate", img180)
    # cv2.waitKey(0)
    output_imagepath = outputpath + 'rotate' + filename
    cv2.imwrite(output_imagepath, img180)

filelist = os.listdir(image_path) #获取主目录下的子文件
for first_dir in filelist:
    img_path = image_path + '/' + first_dir
    #imagename, suffix = os.path.splitext(first_dir)
    rotateimage(img_path, outputpath, first_dir)
