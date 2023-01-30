import os
import nibabel as nib
import numpy as np
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt

from vnet.test.dataloader_test import slice_3Dmatrix, concat_3Dmatrices
from nilearn.image import resample_img

from model import VNet
from file_load import file_read
from vnet.test.dataloader_test import dataset

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

patch = (104, 120, 48)
overlap = (80, 80, 24)


# patch = (32, 104, 48)
# overlap = (20, 80, 24)

def make_dict(img, name):
    dict_array = {
        'img': img,
        'name': name
    }
    return dict_array


def train(model, test_generator, save_pth, device, filename):
    for idx in range(len(test_generator)):
        # image load
        orig_image = test_generator.list[idx][0]
        affine = test_generator.list[idx][1]
        affine_orig = test_generator.list[idx][2]
        shape_orig = test_generator.list[idx][3]

        # list for plot
        ls_rst = []

        ls_rst.append(make_dict(orig_image, 'Original'))

        # cropping
        cr_img = []
        a, b, c = orig_image.shape
        # a, b, c = shape_orig
        cr_img.append(orig_image[0:int((a * 5) / 8), :, :])  # right
        cr_img.append(orig_image[a - int((a * 5) / 8):, :, :])  # left

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

            ## test
            for p in patches:
                model.train()
                p = torch.from_numpy(p).to(device, dtype=torch.float)
                with torch.no_grad():
                    outputs_seg = model.forward(p)
                outputs_seg = F.softmax(outputs_seg, dim=CHANNELS_DIMENSION)
                one_output_seg = outputs_seg.detach().cpu().numpy()[0]  # shape(n_class ,x ,y, z)

                predictions.append(one_output_seg)

            predictions = np.array(predictions)
            orig = []
            for idx in range(3):
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

        prob_bst = np.zeros((a, b, c))
        prob_fgt = np.zeros((a, b, c))

        ### R + L
        # result label
        new[:half_a, :, :] = result_lbl[0][:half_a, :, :]
        new[half_a:, :, :] = result_lbl[1][la - (a - half_a):, :, :]
        ls_rst.append(make_dict(new, 'Result label'))

        # probability map - Breast
        prob_bst[:half_a, :, :] = result_bst[0][0:half_a, :, :]
        prob_bst[half_a:, :, :] = result_bst[1][la - (a - half_a):, :, :]

        # probability map - FGT
        prob_fgt[:half_a, :, :] = result_fgt[0][0:half_a, :, :]
        prob_fgt[half_a:, :, :] = result_fgt[1][la - (a - half_a):, :, :]

        vol_ls = cal_vol(new, 3, affine)
        # show_slice(ls_rst, c//2)

        # save_nii(orig_image, affine, save_pth, 'Orig_T1.nii.gz')
        # save_nii(new, affine, save_pth, 'T1_pre_label.nii.gz')
        # save_nii(prob_bst, affine, save_pth, 'Prob_BST.nii.gz')
        # save_nii(prob_fgt, affine, save_pth,'Prob_FGT.nii.gz')

        FGT_nii = nib.Nifti1Image(prob_fgt, affine=affine)
        BST_nii = nib.Nifti1Image(prob_bst, affine=affine)
        both_nii = nib.Nifti1Image(new, affine=affine)

        FGT_ori = resample_img(img=FGT_nii, target_affine=affine_orig, target_shape=shape_orig,
                               interpolation='continuous')
        BST_ori = resample_img(img=BST_nii, target_affine=affine_orig, target_shape=shape_orig,
                               interpolation='continuous')
        both_ori = resample_img(img=both_nii, target_affine=affine_orig, target_shape=shape_orig,
                                interpolation='continuous')

        filename_FGT = filename + '_' + 'Prob_FGT.nii.gz'
        filename_BST = filename + '_' + 'Prob_BST.nii.gz'
        filename_BST_T1pre = filename + '_' + 'T1_pre_label.nii.gz'

        save_nii(FGT_ori.get_fdata(), affine_orig, save_pth, filename_FGT)
        save_nii(BST_ori.get_fdata(), affine_orig, save_pth, filename_BST)
        save_nii(both_ori.get_fdata(), affine_orig, save_pth, filename_BST_T1pre)

        return new, prob_bst, prob_fgt


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
    nib.save(nii_img, pth + name)

    # nib.save(nii_img, os.path.join(pth, name))


def cal_vol(nparray, num_cls, affine):
    vol_ls = []
    for c in range(1, num_cls):
        vol_size = np.diag(affine, k=0)[:-1]
        voxel = vol_size[0] * vol_size[1] * vol_size[2]
        pred = (nparray == c).astype('float')
        vol_ls.append(pred.sum() * voxel)

    print('Breast_vol: ', vol_ls[0], 'mm^3', '/ FGT_vol:', vol_ls[1], 'mm^3')

    return vol_ls






device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = VNet(n_channels=1, n_classes=3, n_filters=16, normalization='groupnorm', activation = 'ReLU')
model = model.to(device)
model.load_state_dict(torch.load(r'.\model\vnet_breast_0925.pth', map_location=device))
dir_image = 'D:/Study/Dataset/NACData_VNet/img_lesion/images/'
path_save = 'D:/Study/Dataset/NACData_VNet/FGT_SEG_N0/'
#dir_image = r'D:\Study\Dataset\NACData_VNet\img_lesion\images'
txtname = 'C:/Users/86156/Desktop/txt/path_N5N6.txt'
with open(txtname) as file_object:
    lines = file_object.readlines()
    for line in lines:
        filename = line.split('.')[0]
        niifile = file_read(os.path.join(dir_image, line.split()[0]))
        test_loader = dataset(niifile, mode='test')
        train(model, test_loader, path_save, device, filename)