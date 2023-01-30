import os
import torch
from model import VNet

# from test import test
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vnetdata22 import Vnetdata22
from utils.logger import log
import time
import numpy as np
from scipy import ndimage
from setting import parse_opts
import matplotlib.pyplot as plt
import atexit

'''
Training code for VNet  segmentation
Pretrain
Pami
'''
def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder,sets):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_seg = nn.CrossEntropyLoss(ignore_index=-1)

    print("Current setting is:")
    print("\n\n")
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()

    model.train()
    train_time_sp = time.time()


    Loss_list = []  # 存储每次epoch损失值

    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        #scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))

        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks = batch_data

            if not sets.no_cuda:
                volumes = volumes.cuda()

            optimizer.zero_grad()

            # for name, param in model.named_parameters():
            #     print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)

            volumes = volumes.float()
            out_masks = model(volumes)
            # resize label
            [n, _, d, h, w] = out_masks.shape
            new_label_masks = np.zeros([n, d, h, w])
            for label_id in range(n):
                label_mask = label_masks[label_id]
                [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
                label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
                new_label_masks[label_id] = label_mask

            new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            if not sets.no_cuda:
                new_label_masks = new_label_masks.cuda()

            # calculating loss
            loss_value_seg = loss_seg(out_masks, new_label_masks)
            loss = loss_value_seg
            loss.backward()
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))


            # save model
            # if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
            #     # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
            #     model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
            #     model_save_dir = os.path.dirname(model_save_path)
            #     if not os.path.exists(model_save_dir):
            #         os.makedirs(model_save_dir)
            #
            #     log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
            #     torch.save({
            #         'ecpoch': epoch,
            #         'batch_id': batch_id,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict()},
            #         model_save_path)

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
        sets.v_epoch = epoch
        Loss_list.append(loss_value_seg.cpu().detach())

    scheduler.step()
    draw_loss(Loss_list, total_epochs)
    print('Finished training')

# ---------------------#
# loss可视化
# --------------------#
def draw_loss(Loss_list, epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
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


if __name__ == '__main__':
    global Loss_list
    Loss_list = []
    @atexit.register
    def goodbye():
        draw_loss(Loss_list, sets.v_epoch)
        print("Bye")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VNet(n_channels=1, n_classes=3, n_filters=16, normalization='groupnorm', activation='ReLU')
    checkpoint_path = r'../model/vnet_breast_0925.pth'
    pretrained_model_dict = torch.load(checkpoint_path)    #预训练参数
    model_dict = model.state_dict()                        #VNet模型参数（空）
    state_dict = {k:v for k,v in pretrained_model_dict.items() if k in model_dict.keys()}    ####可以导入的参数
    print('load para')
    print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print('load para done')
    model = model.to(device)

    #setting
    sets = parse_opts()
    img_list = '../dataset/NACData_VNet_hundreds/train_2_0.txt' #训练文本
    batch_size = 8
    sets.num_workers = 4
    pin_memory = True
    epochs = 30
    save_intervals = 5
    save_folder = "../trails/models_2_0/{}".format('VNet')    #模型参数保存路径
    best_acc = 0.0

    sets.input_D = 64
    sets.input_H = 128
    sets.input_W = 128
    sets.phase = 'train'
    sets.no_cuda = False

    #getting data
    data_root = '../dataset/NACData_VNet_hundreds'    #
    image_path = os.path.join(data_root, 'N0')    #
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    training_dataset = Vnetdata22(data_root, img_list, sets)
    data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=sets.num_workers,
                             pin_memory=pin_memory)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    train_steps = len(data_loader)

    # training
    train(data_loader, model, optimizer, scheduler, total_epochs=epochs, save_interval=save_intervals,
          save_folder=save_folder, sets=sets)


