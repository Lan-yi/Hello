import os
import torch
#import SimpleITK as sitk
from vnet.train.model3 import VNet
from dataloader_train import dataset_1121
from torch.utils.data import DataLoader
import torch.optim as optim
import nibabel as nib
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.logger import log
from vnet.test.dataloader_test import slice_3Dmatrix
import atexit

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4
patch = (104, 120, 48)
overlap = (80, 80, 24)
patchessize = 4

def make_dict(img, name):
    dict_array = {
        'img': img,
        'name': name
    }
    return dict_array

def draw_loss(Loss_list):
    epoch = len(Loss_list)
    plt.cla()
    x1 = range(0, epoch)
    y1 = Loss_list
    plt.title('Train loss vs.epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig(".Train_loss.png")
    plt.show()

def cal_dice(pred, label, clss):
    """4,x,y,z
    calculate the dice between prediction and ground truth
    input:
        pred: predicted mask
        label: groud truth
        clss: eg. [0, 1] for binary class
    """
    Ncls = len(clss)
    dices = np.zeros(Ncls)
    [depth, height, width] = pred.shape
    for idx, cls in enumerate(clss):
        # binary map
        pred_cls = np.zeros([depth, height, width])
        pred_cls[np.where(pred == cls)] = 1
        label_cls = np.zeros([depth, height, width])
        label_cls[np.where(label == cls)] = 1
        # cal the inter & conv
        inter = np.sum(pred_cls * label_cls)
        conv = np.sum(pred_cls) + np.sum(label_cls)
        dice = (2.0 * inter) / conv
        if conv == 0:
            dice = 0.0
        dices[idx] = dice
    mean_dice = np.mean(dices, axis=0)
    return mean_dice

def save_nii_pre(imarr, filename, affine):
    imarr = imarr.detach().cpu().numpy()
    affine = affine.detach().cpu().numpy()
    target_affine = affine[0]
    nii_img = nib.Nifti1Image(imarr, affine=target_affine)
    pathname = '../dataset/NACData_VNet_hundreds/txttest_N0/nopre_test30/' + filename + '.nii.gz'
    nib.save(nii_img, pathname)

def save_nii_pre_p(imarr, filename, affine):

    affine = affine.detach().cpu().numpy()
    target_affine = affine[0]
    nii_img = nib.Nifti1Image(imarr, affine=target_affine)
    pathname = '../dataset/NACData_VNet_hundreds/txttest_N0/nopre_test30/' + filename + '.nii.gz'
    nib.save(nii_img, pathname)

def save_nii_pre_2(imarr, filename, affine):
    #四通道转一通道
    imarr = imarr.detach().cpu().numpy()
    affine = affine.detach().cpu().numpy()
    target_affine = affine[0]
    imarr = np.argmax(np.array(imarr), axis=0)
    imarr = imarr.astype('uint8')
    nii_img = nib.Nifti1Image(imarr, affine=target_affine)
    # volume_nifty_re = resample_img(
    #     img=volume_nifty,
    #     target_affine=np.diag([1.5, 1.5, 3]),
    #     interpolation='continuous'
    # )
    pathname = '../dataset/NACData_VNet_hundreds/txttest_N0/nopre_test30/' + filename + '.nii.gz'
    nib.save(nii_img, pathname)

def save_nii_ture(imarr, filename, affine):
    imarr = imarr.detach().cpu().numpy()
    affine = affine.detach().cpu().numpy()
    target_affine = affine[0]

    imarr = imarr.astype('uint8')
    nii_img = nib.Nifti1Image(imarr, affine=target_affine)
    pathname = '../dataset/NACData_VNet_hundreds/txttest_N0/nopre_test30/' + filename + '.nii.gz'
    #array_txt = '../dataset/NACData_VNet_hundreds/txttest_N0/nopre_test30/' + filename + '.txt'
    nib.save(nii_img, pathname)
    #np.savetxt(array_txt, imarr, fmt='float')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.fill_(2)

class DICELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, clss):

        diceloss = cal_dice(pred, label, clss)
        return 1-diceloss

def train(model, train_generator, save_folder, device, total_epochs, optimizer, scheduler):
    loss_seg = nn.CrossEntropyLoss()
    #loss_seg = DICELoss()
    loss_seg = loss_seg.cuda()
    model.train()
    global Loss_list  # 存储每次epoch损失值
    Loss_list = []
    global Loss_list2
    Loss_list2 = []
    loss_1epoch = 0
    clss = 4

    for epoch in range(total_epochs):
        loss = 0
        for batch_id, batch_data in enumerate(train_generator):
            # image load
            orig_image, label_ture, name, orig_affine, affine = batch_data
            print(name)
            orig_image = orig_image.cuda()
            # 分patch
            ls_rst = []
            ls_rst.append(make_dict(orig_image, 'Original'))
            size = orig_image.shape
            orig_imagearray = orig_image[0, 0, :, :, :]
            orig_imagearray = orig_imagearray.cpu().numpy()
            label_turearray = label_ture[0, :, :, :]
            label_turearray = label_turearray.cpu().numpy()
            imgpat = slice_3Dmatrix(orig_imagearray, patch, overlap)
            labpat = slice_3Dmatrix(label_turearray, patch, overlap)
            imgpatches = []
            labpatches = []
            [d, h, w] = patch
            img_ip_batch = np.zeros([patchessize, 1, d, h, w])
            lab_ip_batch = np.zeros([patchessize, d, h, w])
            for i in range(len(imgpat)):
                j = i % patchessize
                a = imgpat[i]
                b = labpat[i]
                img_ip_batch[j, 0] = a
                lab_ip_batch[j] = b
                if j == patchessize-1:
                    imgpatches.append(img_ip_batch)
                    labpatches.append(lab_ip_batch)
                    img_ip_batch = np.zeros([patchessize, 1, d, h, w])
                    lab_ip_batch = np.zeros([patchessize, d, h, w])
            #optimizer.zero_grad()
            ### train
            ## 一次送  4 个patch
            loss_patch = 0
            for i in range(len(imgpatches)):
                optimizer.zero_grad()
                p = torch.from_numpy(imgpatches[i]).to(device, dtype=torch.float)
                outputs_seg = model(p)
                outputs_seg = F.softmax(outputs_seg, dim=CHANNELS_DIMENSION)
                # resize label
                new_label_mask = labpatches[i]
                new_label_mask = torch.tensor(new_label_mask).to(torch.int64)
                new_label_mask = new_label_mask.cuda()
            # [n, _, d, h, w] = outputs_seg.shape
            # new_label_masks = np.zeros([n, d, h, w])
            # for label_id in range(n):
            #     label_mask = label_ture[label_id]
            #     [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
            #     label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
            #     scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
            #     label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
            #     new_label_masks[label_id] = label_mask            #
            # new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            # new_label_masks = new_label_masks.cuda()
                # calculating loss

                loss_value_seg = loss_seg(outputs_seg, new_label_mask)    #########CE
                ###diceloss
                # pred = outputs_seg.detach().cpu().numpy()
                # pred = np.argmax(np.array(pred), axis=0)
                # outputs_seg = torch.tensor(pred).to(torch.int64)
                # loss_value_seg = loss_seg(outputs_seg, new_label_mask, range(clss)) #返回四个样本四类共16个值的平均DICE
                ##
                # if name[0] == '../dataset/NACData_VNet_hundreds\\N0\\2845573_20200402_N0.nii.gz':
                #     [ps, c, d, h, w] = outputs_seg.shape
                #     for j in range(ps):
                #         filename = str(epoch) + '_' + '2845573_20200402_pre' + str(i) + str(j)
                #         save_nii_pre(p[j][0], filename + 'p', affine)
                #         save_nii_pre_2(outputs_seg[i], filename, affine)
                #     [ps, d, h, w] = new_label_mask.shape
                #     for k in range(ps):
                #         filename = str(epoch) + '_' + '2845573_20200402_ture' + str(i) + str(k)
                #         save_nii_ture(new_label_mask[k], filename, affine)
                ##
                loss_value_seg.backward()
                loss_patch = loss_patch + loss_value_seg
                optimizer.step()
            temp_loss = loss_patch/len(imgpatches)
            loss += temp_loss
            temp_loss = temp_loss.cpu().detach()
            log.info('Batch: {}-{}, loss = {:.3f}'.format(epoch, batch_id, temp_loss.item()))
            Loss_list2.append(temp_loss)  #########

        scheduler.step()
        loss_1epoch = loss/len(train_generator.dataset.path)
        loss_1epoch = loss_1epoch.cpu().detach()
        log.info('Batch: {}, loss = {:.3f}'.format(epoch, loss_1epoch.item()))

        #log.info('Batch: {}-{}, loss = {:.3f}, loss_seg = {:.3f}'.format(epoch, batch_id, loss.item(), loss_value_seg.item()))

        loss_array = np.array(loss_1epoch)
        Loss_list.append(loss_array)
        # save model
        model_save_path = '{}_epoch_{}.pth.tar'.format(save_folder, epoch)
        model_save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        log.info('Save checkpoints: epoch = {}'.format(epoch))
        torch.save({
            'ecpoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            model_save_path)

    draw_loss(Loss_list2)
    draw_loss(Loss_list)
    print('Finished training')

if __name__ == '__main__':
    @atexit.register
    def goodbye():
        draw_loss(Loss_list2)
        print("Bye")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VNet(n_channels=1, n_classes=4, n_filters=16, normalization='groupnorm', activation='ReLU')    #
    model.apply(weights_init)
    print(model)

    #model.load_state_dict(torch.load(r'..\model\vnet_breast_0925.pth', map_location=device))    #预测需要的模型参数

    # checkpoint_path = r'..\model\vnet_breast_0925.pth'
    # pretrained_model_dict = torch.load(checkpoint_path)    #预训练参数
    model_dict = model.state_dict()                        #VNet模型参数（空）
    # state_dict = {k:v for k,v in pretrained_model_dict.items() if k in model_dict.keys()}    ####可以导入的参数
    # print('load para')
    # print(state_dict.keys())
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    # print('load para done')
    model = model.to(device)
    dir_image = r'../dataset/NACData_VNet_hundreds'    #根目录
    txtname = '../dataset/NACData_VNet_hundreds/train_3_56.txt'
    save_folder = "../trails/models_w356/{}".format('VNet')
    epochs = 10
    imglist = []
    with open(txtname) as file_object:
        lines = file_object.readlines()
        lines = [line.strip() for line in lines]
        imglist = lines
    train_loader = dataset_1121(dir_image, imglist, mode='train')
    data_loader = DataLoader(train_loader, batch_size=1, shuffle=True, num_workers=0,
                                 pin_memory=True)
    #SGD
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    #Adam
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train(model, data_loader, save_folder, device, epochs, optimizer, scheduler)







