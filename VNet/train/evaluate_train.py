import os
import nibabel as nib
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from utils.logger import log
from vnet.test.dataloader_test import slice_3Dmatrix
import atexit
import math

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

patch = (104, 120, 48)
overlap = (80, 80, 24)

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

def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

# class SoftDiceLoss(nn.Module):
#     def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
#         super(SoftDiceLoss, self).__init__()
#
#         self.do_bg = do_bg
#         self.batch_dice = batch_dice
#         self.apply_nonlin = apply_nonlin
#         self.smooth = smooth
#
#     def forward(self, x, y, loss_mask=None):
#         shp_x = x.shape
#
#         if self.batch_dice:
#             axes = [0] + list(range(2, len(shp_x)))
#         else:
#             axes = list(range(2, len(shp_x)))
#
#         if self.apply_nonlin is not None:
#             x = self.apply_nonlin(x)
#
#         tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, False)
#
#         nominator = 2 * tp + self.smooth
#         denominator = 2 * tp + fp + fn + self.smooth
#
#         dc = nominator / (denominator + 1e-8)
#
#         if not self.do_bg:
#             if self.batch_dice:
#                 dc = dc[1:]
#             else:
#                 dc = dc[:, 1:]
#         dc = dc.mean()
#
#         return 1-dc

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)
        print(dc)
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1-dc

def train(model, train_generator, save_folder, device, total_epochs, optimizer, scheduler):
    loss_seg = nn.CrossEntropyLoss()
    #loss_seg = SoftDiceLoss(**{'apply_nonlin': softmax_helper, 'batch_dice': True, 'smooth': 1e-5, 'do_bg': True})
    loss_seg = loss_seg.cuda()
    model.train()
    global Loss_list
    Loss_list = []        # 存储每次epoch损失值
    global Loss_list2
    Loss_list2 = []
    loss = 0
    loss_1epoch = 0

    for epoch in range(total_epochs):
        count = 0
        #for idx in range(len(train_generator)):
        for batch_id, batch_data in enumerate(train_generator):
            # image load
            orig_image, label_ture, name, orig_affine, affine = batch_data
            print(name)
            orig_image = orig_image.cuda()
            optimizer.zero_grad()

            # list for plot
            ls_rst = []
            ls_rst.append(make_dict(orig_image, 'Original'))

            size = orig_image.shape

            # cr_img = torch.from_numpy(cr_img).to(device, dtype=torch.float)
            # outputs_seg = model.forward(cr_img)

            orig_imagearray = orig_image[0, 0, :, :, :]
            orig_imagearray = orig_imagearray.cpu().numpy()
            label_turearray = label_ture[0, :, :, :]
            label_turearray = label_turearray.cpu().numpy()
            imgpat = slice_3Dmatrix(orig_imagearray, patch, overlap)
            labpat = slice_3Dmatrix(label_turearray, patch, overlap)
            imgpatches = []
            labpatches = []
            for i in range(len(imgpat)):
                img_ip = np.array(imgpat[i])[np.newaxis][np.newaxis]
                iab_ip = np.array(labpat[i])[np.newaxis][np.newaxis]
                imgpatches.append(img_ip)
                labpatches.append(iab_ip)
            #array to tensor

            ## train
            loss_patch = 0
            for i in range(len(imgpatches)):
                optimizer.zero_grad()
                p = torch.from_numpy(imgpatches[i]).to(device, dtype=torch.float)
                outputs_seg = model(p)

            #outputs_seg = F.softmax(outputs_seg, dim=CHANNELS_DIMENSION)
            #one_output_seg = outputs_seg.detach().cpu().numpy()[0]  # shape(n_class ,x ,y, z)

            # resize label
                new_label_mask = labpatches[i][:, 0, :, :, :]
                # new_label_mask = labpatches[i]  ##########0107
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
            #     new_label_masks[label_id] = label_mask
            #
            # new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            # new_label_masks = new_label_masks.cuda()
                # calculating loss
                loss_value_seg = loss_seg(outputs_seg, new_label_mask)
                loss_value_seg.backward()
                loss_patch = loss_patch + loss_value_seg
                optimizer.step()
            temp_loss = loss_patch/len(imgpatches)
            loss += temp_loss
            temp_loss = temp_loss.cpu().detach()
            log.info('Batch: {}-{}, loss = {:.3f}'.format(epoch, batch_id, temp_loss.item()))
            Loss_list2.append(temp_loss)  #########

        loss_1epoch = loss / len(train_generator.dataset.path)
        loss_1epoch = loss_1epoch.cpu().detach()
        scheduler.step(loss_1epoch)
        log.info('Batch: {}, loss = {:.3f}'.format(epoch, loss_1epoch.item()))
        loss_array = np.array(loss_1epoch)
        Loss_list.append(loss_array)
        if epoch > 0 and abs(Loss_list[epoch] - Loss_list[epoch-1]) < 0.01:
            count = count+1
        if count == 3:
            quit()

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

    draw_loss(Loss_list)
    x_len = len(Loss_list2)
    draw_loss(Loss_list2)
    print('Finished training')


@atexit.register
def goodbye():
    draw_loss(Loss_list)
    print("Bye")
# plot center slice
def show_slice(imgls, slice):
    cnt = len(imgls)
    plt.figure(dpi=150)
    for idx, img in enumerate(imgls):
        plt.subplot(1, cnt, idx + 1)
        if img['name'] == 'original':
            plt.imshow(np.rot90(img['img'][:, :, slice]), cmap='gray')
        else:
            plt.imshow(np.rot90(img['img'][:, :, slice]), cmap='gray')
        plt.title(img['name'])
        plt.axis('off')
    plt.show()

def save_nii(imarr, affine, pth, name):
    nii_img = nib.Nifti1Image(imarr, affine=affine)
    # nib.save(nii_img, pth + name)
    nib.save(nii_img, os.path.join(pth, name))

def cal_vol(nparray, num_cls, affine):
    vol_ls = []
    for c in range(1, num_cls):
        vol_size = np.diag(affine, k=0)[:-1]
        voxel = vol_size[0] * vol_size[1] * vol_size[2]
        pred = (nparray == c).astype('float')
        vol_ls.append(pred.sum() * voxel)

    print('Breast_vol: ', vol_ls[0], 'mm^3', '/ FGT_vol:', vol_ls[1], 'mm^3')

    return vol_ls
















