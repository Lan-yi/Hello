import numpy as np
import cv2
import  os
from shutil import copy, rmtree
import random

# 首先以灰色读取一张照片
src = cv2.imread("D:/Pami/Dataset/fenlei/192297_T3_N6_26.tiff", 0)
# 然后用ctvcolor（）函数，进行图像变换。
src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
# 显示图片
cv2.imshow("input", src)
cv2.imshow("output", src_RGB)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cwd = os.getcwd()
# data_root = os.path.join(cwd, "medicaldata")
# origin_flower_path = os.path.join(data_root, "medicalphotos")
# flower_class = [cla for cla in os.listdir(origin_flower_path)
#                     if os.path.isdir(os.path.join(origin_flower_path, cla))]
# for cla in flower_class:
#         cla_path = os.path.join(origin_flower_path, cla)
#         images = os.listdir(cla_path)
#         num = len(images)
#         for image in enumerate(images):
#             image_path = "D:/Python/PythonProject/Hello/Classification/data_set/medicaldata/"
#             imagrtu = cv2.imread(image_path+image)
#             imagrtu = cv2.cvtColor(imagrtu, cv2.COLOR_GRAY2BGR)
#
#
#
