import cv2
import numpy as np
import os
#import natsort


dir = 'test/nPCR/'
directory = 'D:/Study/Dataset/medicaldata_uint16_png_3_2_2/' + dir
#directory = '../data_set/medicaldata_uint16_png_3_2_2/' + dir
files = os.listdir(directory)
#files = natsort.natsorted(files)

def change16_8(file,filename):
    img = cv2.imread(file,-1)
    imgarray = np.array(img)
    high = np.max(imgarray)
    low = np.min(imgarray)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (imgarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
    imgarray = (newimg * 255).astype('uint8')
    output_path = 'D:/Study/Dataset/imgshow/' + dir + filename
    cv2.imwrite(output_path, imgarray)


for i in range(len(files)):
    file = directory+ '/' + files[i]    #dcm2D文件名
    change16_8(file,files[i])
    print(file)





