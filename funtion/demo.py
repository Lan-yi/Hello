import os
import random #乱序使用随机函数
import shutil

# ---------------------------------#
#对文件夹下文件重命名
# ---------------------------------#
def renamefile_addmask(file_path):
    fileslist = os.listdir(file_path) #获取主目录下的子文件列表
    for first_dir in fileslist:
        old_filename = file_path + '/' + first_dir  #真正的指向了文件
        New_FN = first_dir[:-4] + '_mask' + first_dir[-4:]
        old_filename = file_path +  '/' + first_dir
        New_File_Name = file_path +  '/' + New_FN
        os.renames(old_filename, New_File_Name)

#############image
file_path = '/UNet/UNet-pytorch-master/data/8/train/masks'
renamefile_addmask(file_path)
