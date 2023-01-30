import os
import torch
from model import VNet

from file_load import file_read
from vnet.test.dataloader_test import dataset
# from test import test
from torch import nn
import torch.optim as optim
from torchvision import datasets

'''
Training code for VNet  segmentation
Pami
'''
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VNet(n_channels=1, n_classes=3, n_filters=16, normalization='groupnorm', activation = 'ReLU')
print(model)
model = model.to(device)
dir_image = 'D:/Study/Dataset/NACData_VNet/img_lesion/images/'

#getting data
data_root = './dataset/NACData_VNet'
image_path = os.path.join(data_root, "images")
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, ""))
train_num = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1, shuffle=True,
                                               num_workers=0)




txtname = 'C:/Users/86156/Desktop/txt/path_N5N6.txt'
with open(txtname) as file_object:
    lines = file_object.readlines()
    for line in lines:
        filename = line.split('.')[0]
        niifile = file_read(os.path.join(dir_image, line.split()[0]))
        test_loader = dataset(niifile, mode='test')
        new, prob_bst, prob_fgt = train(model, train_loader, path_save, device, filename)



# define loss function
    loss_function = nn.CrossEntropyLoss()

# construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    epochs = 50
    best_acc = 0.0
    save_path = './Vet.pth'
    train_steps = len(train_loader)






















