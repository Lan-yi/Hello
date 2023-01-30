import os

import torch
# print(torch.__version__)

#import SimpleITK as sitk

from vnet.train.model import VNet

from file_load import file_read
from vnet.test.dataloader_test import dataset
# from test import test
from evaluate2 import test2   #

#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VNet(n_channels=1, n_classes=3, n_filters=16, normalization='groupnorm', activation='ReLU')

model.load_state_dict(torch.load(r'.\model\vnet_breast_0925.pth', map_location=device))    #预测需要的模型参数
#model.load_state_dict(torch.load(r'.\trails\models_2_0\VNet_epoch_29.pth.tar')['state_dict'])    #预测需要的模型参数

# checkpoint_path = r'.\trails\models_3\VNet_epoch_19_batch_0.pth.tar'
# pretrained_model_dict = torch.load(checkpoint_path)    #预训练参数
# model_dict = model.state_dict()                        #VNet模型参数（空）
# state_dict = {k:v for k,v in pretrained_model_dict.items() if k in model_dict.keys()}    ####可以导入的参数
# print('load para')
# print(state_dict.keys())
# model_dict.update(state_dict)
# model.load_state_dict(model_dict)
# print('load para done')

model = model.to(device)

dir_image = r'../dataset/NACData_VNet_hundreds'
path_save = r'../dataset/NACData_VNet_hundreds/txttest_N0/N0_uint8_CE'  #预测标注结果
txtname = './dataset/NACData_VNet_hundreds/test.txt'
count = 0
labelfile = []
with open(txtname) as file_object:
    lines = file_object.readlines()
    for line in lines:
        line = line.strip()
        filename = line.split('.')[0]
        labelname = os.path.join(dir_image, line.split(' ')[1])    #
        niifile = file_read(os.path.join(dir_image, line.split()[0]))
        test_loader = dataset(niifile, mode='test')
        #new, prob_bst, prob_fgt = test(model, test_loader, path_save, device, filename)
        test2(model, test_loader, path_save, device, filename, labelname)
        count = count + 1
        print(count, '/', len(lines))




# one test
# file_path = r'.\dataset\NACData_VNet_hundreds'        #预测文件目录
# niifile = file_read(os.path.join(file_path, r'N0\192297_20181027_N0.nii.gz'))
#
# test_loader = dataset(niifile, mode = 'test')
# print(len(test_loader))
# path_save = os.path.join(file_path, 'onetest')
# filename = '192297_20181027_N0'
# if not os.path.isdir(path_save):
#     os.makedirs(path_save)
# new, prob_bst, prob_fgt = test(model, test_loader, path_save, device, filename)
