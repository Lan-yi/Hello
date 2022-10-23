import cv2
#import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#import imageio
#import SimpleITK as sitk
#import accimage


#file_path = "D:/Study/ImageData/mohu/image/origin/SmoothIm3.2ModelInv.bmp"
#file_path = "D:/Study/ImageData/mohu/label/origin/Im3.2ModelInv.bmp"

#tiff_path = 'D:/Image/tower.jpg'
tiff_path = '/Classification/data_set/medicaldata_uint16_png/192297_T3_N6_26_uint16.png'
#tiff_path = 'D:/Pami/Dataset/PCR_nPCR/tiff_tiled_planar_lzw.tiff'

# imgcv = cv2.imread(tiff_path)
#
# height, width, channel = imgcv.shape
# size_decrease = (512,512)
# img = cv2.resize(imgcv,size_decrease,interpolation=cv2.INTER_CUBIC)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray",gray)
#cv2.waitKey(0)
#high_gray = np.max(gray)

#imgarray = np.array(gray)

#imgarray = imgarray.astype("uint8")
# high_imgarray = np.max(imgarray)
#
# for i in range(0,511):
#     for j in range(0, 511):
#         if imgarray[i][j] > 0:
#             imgarray[i][j] = 1
#
# cv2.imshow("Gray",imgarray)
# cv2.waitKey(0)




# high = np.max(imgarray)
# low = np.min(imgarray)
# lungwin = np.array([low * 1., high * 1.])
# newimg = (imgarray - lungwin[0]) / (lungwin[1] - lungwin[0])    # 归一化
# imgarray = (newimg * 1000).astype('float64')  # GE将像素值扩展到[0,3000]，PHI[0.5500]


# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread(tiff_path,-1)
# imgarray = np.array(img)
# cv2.namedWindow("Image")
# cv2.imshow("Image",img)
# cv2.waitKey(0)#释放窗口
# cv2.destroyAllWindows()

# import numpy as np

#
# def load(path):
#     img = imageio.imread(path)
#     return img


# img = cv2.imread(tiff_path,-1)
# img = np.array(img).astype(np.uint8)#假设对图像做仿射变换
# img = Image.fromarray(img)
# img.save('new_img.tiff')
# plt.imshow(img)
# plt.show()



###########PIL
# img = Image.open(tiff_path)
# imgarray = np.array(img)
# img.show()






