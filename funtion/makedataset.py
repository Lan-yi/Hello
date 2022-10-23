import os
import random #乱序使用随机函数
import shutil

# ---------------------------------#
#制作random数据集
# ---------------------------------#
def make_randomdataset(images_filedir,masks_filedir,traindir,testdir):
    imageslist = os.listdir(images_filedir) #获取主目录下的子文件列表
    maskslist = os.listdir(masks_filedir)
    count = 0
    for first_dir in imageslist:
        count = count + 1
        image_path = images_filedir + '/' + first_dir  #真正的指向了文件
        mask_path = masks_filedir + '/' + first_dir  #真正的指向了文件

        if count % 10 == 0:#定义为每10张图片，取一张为测试图片
            image_testdir = testdir + 'images/' + first_dir
            mask_testdir = testdir + 'masks/' + first_dir
            shutil.copy(image_path,image_testdir)     #image复制到测试集
            shutil.copy(mask_path,mask_testdir)     #对应mask复制到测试集

        else:
            image_traindir = traindir + 'images/' + first_dir
            mask_traindir = traindir + 'masks/' + first_dir
            shutil.copy(image_path, image_traindir)  # image复制到训练集
            shutil.copy(mask_path, mask_traindir)  # 对应mask复制到训练集



#############image
images_filedir = 'D:/Study/ImageData/mohu/rename_dataset/images'
masks_filedir = 'D:/Study/ImageData/mohu/rename_dataset/masks'

traindir = 'D:/Study/ImageData/mohu/random/train/'
testdir = 'D:/Study/ImageData/mohu/random/test/'
make_randomdataset(images_filedir,masks_filedir,traindir,testdir)




#保存地址
# first_filedir = "D:/Study/Python/PythonProject/Hello/UNet/unet-pytorch-main/VOCdevkit/VOC2007/ImageSets/Segmentation/"
# traindir = first_filedir + "train.txt"
# evaldir = first_filedir + "val.txt"
# #判断该地址下，是否含有已经存在的txt
# if(os.path.exists(traindir)):  # 判断有误文件
#     os.remove(traindir)  # 删除文件
# if(os.path.exists(evaldir)):  # 判断有误文件
#     os.remove(evaldir)  # 删除文件
# fp = open(traindir,'w')
# fp = open(evaldir,'w')
# #构建生成txt文件函数，传入参数为（文件夹地址，训练txt地址，测试txt地址）
# def make_txt(first_filedir,traindir,evaldir):
#     #构建训练列表
#     train_list=[]
#     eval_list=[]
#     filelist = os.listdir(first_filedir) #获取主目录下的子文件如N0，N1
#     for first_dir in filelist:
#         second_filedir = first_filedir+'/'+first_dir# 遍历到/home/aistudio/ttst/N0
#         second_filelist = os.listdir(second_filedir)# 读取主目录下的子文件如N0_NPY
#         count = 0#计数
#         for second_dir in second_filelist:
#             count = count + 1
#             third_filedir=second_filedir+'/'+second_dir#真正的指向了文件N0_0_21.npy文件
#             if count % 10 == 0:#定义为每10张图片，取一张为测试图片
#                 #此处可以自定义*
#                 eval_list.append(third_filedir + ' ' + second_dir[1] + '\n')
#             else:
#                 train_list.append(third_filedir + ' ' + second_dir[1] + '\n')
#     #乱序
#     random.shuffle(eval_list)
#     #读写
#     with open(evaldir, 'a') as f:
#         for eval_image in eval_list:
#             f.write(eval_image)
#     #乱序
#     random.shuffle(train_list)
#     with open(traindir, 'a') as f:
#         for train_image in train_list:
#             f.write(train_image)
# #执行函数
# make_txt(first_filedir,traindir,evaldir)




