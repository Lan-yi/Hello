import os
import torch
#import SimpleITK as sitk
from vnet.train.model3 import VNet
from file_load import file_read
from dataloader_train import dataset_1121

# from test import test
from evaluate_train import train
from torch.utils.data import DataLoader
import torch.optim as optim
import atexit
from torch import nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.fill_(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VNet(n_channels=1, n_classes=4, n_filters=16, normalization='groupnorm', activation='ReLU')    #

model.apply(weights_init)
#model.load_state_dict(torch.load(r'..\model\vnet_breast_0925.pth', map_location=device))    #预测需要的模型参数

checkpoint_path = r'..\trails\double\models_w356_double_c_pre_mask\VNet_epoch.pth.tar'
pretrained_model_dict = torch.load(checkpoint_path)    #预训练参数
model_dict = model.state_dict()                        #VNet模型参数（空）
state_dict = {k:v for k,v in pretrained_model_dict.items() if k in model_dict.keys()}    ####可以导入的参数
print('load para')
print(state_dict.keys())
model_dict.update(state_dict)
model.load_state_dict(model_dict)
print('load para done')

model = model.to(device)
print(model)

dir_image = r'../dataset/NACData_VNet_hundreds'    #根目录
txtname = '../dataset/NACData_VNet_hundreds/test_mask.txt'
save_folder = "../trails/double/models_w356_double_c_pre_mask/{}".format('VNet')
epochs = 50
count = 0
imglist = []

with open(txtname) as file_object:
    lines = file_object.readlines()
    lines = [line.strip() for line in lines]
    imglist = lines

train_loader = dataset_1121(dir_image, imglist, mode='train')
data_loader = DataLoader(train_loader, batch_size=1, shuffle=True, num_workers=0,
                             pin_memory=True)
#SGD
#optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.99)
#Adam
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-06, eps=1e-08)

train(model, data_loader, save_folder, device, epochs, optimizer, scheduler)
count = count + 1
print(count, '/', len(lines))





#分patch，没用DataLoader
# count = 0
# imglist = []
# labellist = []
# with open(txtname) as file_object:
#     lines = file_object.readlines()
#     lines = [line.strip() for line in lines]
#     for line in lines:
#         filename = line.split('.')[0]
#         imglist.append(os.path.join(dir_image, line.split()[0]))
#         labellist.append(os.path.join(dir_image, line.split()[1]))
#     niifile = file_read(imglist, labellist)
#     #train_loader含img和label两个数据的列表
#     train_loader = dataset(niifile, mode='train')    #bz?
#         #new, prob_bst, prob_fgt = test(model, test_loader, path_save, device, filename)
#     train(model, train_loader, path_save, device, filename, epochs)
#     count = count + 1
#     print(count, '/', len(lines))