import os
import natsort
import cv2
import numpy as np
#---------------------#
#按window默认规则读取文件夹下文件
#输出灰度最值
#--------------------#

folder = 'D:/Study/Python/PythonProject/Hello/Classification/data_set/medicaldata_uint16_png_3_2_3/train/PCR'
files = os.listdir(folder)
files = natsort.natsorted(files)
i = 0
for file in files:
    filename = folder + '/' + files[i]
    img = cv2.imread(filename, -1)
    imgarray = np.array(img)
    maxvalue = str(np.max(imgarray))
    minvalue = str(np.min(imgarray))
    #print(files[i] + ' : ' + minvalue + '-' + maxvalue)
    print(files[i] + ' : ' +maxvalue)
    # a = file[-8:-7]
    # if file[-8:-7] == '6':
    #     print(files[i] + ' : ' +maxvalue)
    i = i + 1
