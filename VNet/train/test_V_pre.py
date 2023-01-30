import os
import torch
from vnet.train.model import VNet

from file_load import file_read
from vnet.test.dataloader_test import dataset
# from test import test
from vnet.test.dataloader_test import slice_3Dmatrix, concat_3Dmatrices
from torch.utils.data import DataLoader
from vnet.train.vnetdata22 import Vnetdata22
import numpy as np
from vnet.train.setting import parse_opts
from nilearn.image import resample_img
import torch.nn.functional as F
import nibabel as nib
from vnet.train.utils.file_process import load_lines
'''
test code for VNet  segmentation
Pretrain
Pami
'''
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

def seg_eval(pred, label, clss):
    """
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
        s = pred_cls + label_cls
        inter = len(np.where(s >= 2)[0])
        conv = len(np.where(s >= 1)[0]) + inter
        try:
            dice = 2.0 * inter / conv
        except:
            print("conv is zeros when dice = 2.0 * inter / conv")
            dice = -1
        dices[idx] = dice
    return dices

def save_nii(imarr, affine, pth, name):
    nii_img = nib.Nifti1Image(imarr, affine = affine)
    #nib.save(nii_img, pth + name)
    nib.save(nii_img, os.path.join(pth, name))

def test(data_loader, model, img_names, sets):
    news = []
    save_pth= r'.\dataset\NACData_VNet_hundreds\txttest_N0\N0_1117'
    model.eval()  # for testing
    for batch_id, batch_data in enumerate(data_loader):
        ###########
        niifile = file_read(os.path.join(sets.data_root, img_names[batch_id]))
        test_loader = dataset(niifile, mode='test')
        data = test_loader.list[0][0]
        affine = test_loader.list[0][1]
        affine_orig = test_loader.list[0][2]
        shape_orig = test_loader.list[0][3]

        #img = nib.load(os.path.join(sets.data_root, img_names[batch_id]))
        #data = img.get_fdata()
        #affine_orig = img.affine
        #shape_orig = data.shape
        # list for plot
        ls_rst = []
        ls_rst.append(make_dict(data, 'Original'))
        # cropping
        cr_img = []
        a, b, c = data.shape
        # a, b, c = shape_orig
        cr_img.append(data[0:int((a * 5) / 8), :, :])  # right
        cr_img.append(data[a - int((a * 5) / 8):, :, :])  # left
        result_lbl = []  # result_label
        result_bst = []  # prob_breast
        result_fgt = []  # prob_FGT
        for image in cr_img:
            size = image.shape
            pat = slice_3Dmatrix(image, patch, overlap)

            patches = []
            predictions = []

            for i in range(len(pat)):
                img_ip = np.array(pat[i])[np.newaxis][np.newaxis]
                patches.append(img_ip)

            for p in patches:
                model.eval()    #### model.train()
                p = torch.from_numpy(p).to(device, dtype = torch.float)
                with torch.no_grad():
                    outputs_seg = model.forward(p)
                outputs_seg = F.softmax(outputs_seg, dim=CHANNELS_DIMENSION)
                one_output_seg = outputs_seg.detach().cpu().numpy()[0]                  # shape(n_class ,x ,y, z)

                predictions.append(one_output_seg)
            predictions = np.array(predictions)
            orig = []
            for idx in range(3):  # 4目标得改这？
                each = concat_3Dmatrices(predictions[:, idx, ...], size, patch, overlap)
                orig.append(each)

            result_bst.append(np.array(orig)[1])
            result_fgt.append(np.array(orig)[2])

            orig = np.argmax(np.array(orig), axis=0)
            orig = orig.astype('uint8')

            result_lbl.append(orig)
        ## SAVE ##
        ra, rb, rc = result_lbl[0].shape
        la, lb, lc = result_lbl[1].shape

        half_a = int(ra * (4 / 5))

        new = np.zeros((a, b, c))
        new = new.astype('uint8')  #
            ### R + L
            # result label
        new[:half_a, :, :] = result_lbl[0][:half_a, :, :]
        new[half_a:, :, :] = result_lbl[1][la - (a - half_a):, :, :]
        ls_rst.append(make_dict(new, 'Result label'))
        both_nii = nib.Nifti1Image(new, affine=affine)
        both_ori = resample_img(img=both_nii, target_affine=affine_orig, target_shape=shape_orig,
                                    interpolation='continuous')
        filename = img_names[batch_id].split('.')[0]
        filename_BST_T1pre = filename + '_' + 'T1_pre_label.nii.gz'

        print(save_pth, filename_BST_T1pre)
        if os.path.exists(os.path.join(save_pth, filename_BST_T1pre)):
            print('file exict')
            continue
        save_nii(both_ori.get_fdata(), affine_orig, save_pth, filename_BST_T1pre)
        news.append(new)
    return news

# save seg result
def save_segresult(ori_label, result, result_filepath):
    result_filename = result_filepath.split('\\')[1]
    # 将仿射矩阵和头文件保存下来
    affine = ori_label.affine
    hdr = ori_label.header
    # 形成新的nii文件
    new_nii = nib.Nifti1Image(result, affine, hdr)
    nib.save(new_nii, './dataset/test/seg_result/' + result_filename)

if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'
    sets.gpu_id = '0'

    ##################
    sets.model_depth = 10
    sets.resnet_shortcut = 'B'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sets.n_seg_classes = 3

    # getting 4class model
    # sets.resume_path = "./trails/models/VNet_epoch_99.pth.tar"
    # checkpoint = torch.load(sets.resume_path)
    # net = VNet(n_channels=1, n_classes=4, n_filters=16, normalization='groupnorm', activation='ReLU')
    # net.load_state_dict(checkpoint['state_dict'])
    # net = net.to(device)

    # getting 3class model
    net = VNet(n_channels=1, n_classes=sets.n_seg_classes, n_filters=16, normalization='groupnorm', activation='ReLU')
    net.load_state_dict(torch.load("./model/vnet_breast_0925.pth"))
    net = net.to(device)

    # dataset setting
    sets.img_list = r'.\dataset\NACData_VNet_hundreds\test_2_0.txt'
    sets.data_root = './dataset/NACData_VNet_hundreds'

    # data tensor
    testing_data = Vnetdata22(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # testing
    img_names = [info.split(" ")[0] for info in load_lines(sets.img_list)]
    masks = test(data_loader, net, img_names, sets)

    # evaluation: calculate dice
    # label_names = [info.split(" ")[1] for info in load_lines(sets.img_list)]
    # Nimg = len(label_names)
    # dices = np.zeros([Nimg, sets.n_seg_classes])
    # for idx in range(Nimg):
    #     label = nib.load(os.path.join(sets.data_root, label_names[idx]))
    #     #save_segresult(label, masks[idx], label_names[idx])
    #     label = label.get_data()
    #     #dices[idx, :] = seg_eval(masks[idx], label, range(sets.n_seg_classes))

    # print result
    # for idx in range(Nimg):
    #     for c in range(sets.n_seg_classes):
    #         print('dice for {} class-{} is {}'.format(label_names[idx], c, dices[idx][c]))

        # mean_dice_per_task = np.mean(dices[:, idx])
        # print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))