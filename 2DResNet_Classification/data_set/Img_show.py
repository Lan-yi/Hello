import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#file_dcm =  'D:/Study/Dataset/GE/output/2898346_T3_N6_49.png'

# file_dcm =  filepath + '2693114_T3_N6_39.png'
#file_dcm = "medicaldata_uint16_png_3_2/val/nPCR/1106817_T3_N6_28.png"
#file_dcm = "D:/Study/Dataset/GE/2496179_T3_N5_23.png"
file_dcm = "../data_set/medicaldata_uint16_png_3_2_3/test/PCR/3068534_T3_N6_39.png"

# img = cv2.imread(file_dcm,-1)
# imgarray = np.array(img)
#
# print('maxvalue:' + str(np.max(imgarray)))
# print('minvalue:' + str(np.min(imgarray)))
#
#
# high = np.max(imgarray)
# low = np.min(imgarray)
# lungwin = np.array([low * 1., high * 1.])
# newimg = (imgarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
# imgarray = (newimg * 255).astype('uint8')  # GE将像素值扩展到[0,3000]，PHI[0.5500]
#
# output_dcm = '../data_set/medicaldata_uint16_png_3_2/1106817_T3_N6_26.png'
# cv2.imwrite(output_dcm, imgarray)

##################PIL显示
# image = Image.open(file_dcm).convert('RGB')
# imgarray = np.array(image)
# print('maxvalue:' + str(np.max(imgarray)))
#
# high = np.max(imgarray)
# low = np.min(imgarray)
# lungwin = np.array([low * 1., high * 1.])
# newimg = (imgarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
# imgarray = (newimg * 255).astype('uint8')
#
# plt.imshow(imgarray)
# plt.show()

#############cv2显示
img = cv2.imread(file_dcm,-1)
imgarray = np.array(img)

print('maxvalue:' + str(np.max(imgarray)))
print('minvalue:' + str(np.min(imgarray)))


high = np.max(imgarray)
low = np.min(imgarray)
lungwin = np.array([low * 1., high * 1.])
newimg = (imgarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
imgarray = (newimg * 255).astype('uint8')  # GE将像素值扩展到[0,3000]，PHI[0.5500]

# output_dcm = '../data_set/medicaldata_uint16_png_3_2/2170906_T3_N6_29.png'
# cv2.imwrite(output_dcm, imgarray)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", imgarray)
cv2.waitKey(0)
cv2.destroyAllWindows()

##################转三通道############################
# filepath =  'D:/Study/Python/PythonProject/Hello/Classification/data_set/medicaldata_uint16_png/'
# file_dcm =  filepath + '192297_T3_N6_26_uint16.png'
#
# img16 = cv2.imread(file_dcm,-1)
# imgarray16 = np.array(img16)
#
#
# img = Image.open(file_dcm).convert('RGB')
# imgarray = np.array(img)
# imgarray = imgarray.astype('uint16')
# imgarray[:,:,0] = imgarray16
# imgarray[:,:,1] = imgarray16
# imgarray[:,:,2] = imgarray16
# output_dcm = filepath + '192297_T3_N6_26_uint16_out.png'
# cv2.imwrite(output_dcm, imgarray)
#
# # high = np.max(imgarray)
# # low = np.min(imgarray)
# # lungwin = np.array([low * 1., high * 1.])
# # newimg = (imgarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
# # imgarray = (newimg * 255).astype('uint8')  # GE将像素值扩展到[0,3000]，PHI[0.5500]
#
#
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.imshow("image", imgarray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()







