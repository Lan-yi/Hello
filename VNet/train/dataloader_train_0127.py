'''
many nifti file load.
'''

import nibabel as nb
import os
import sys
import numpy as np
import math
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from nilearn.image import resample_img
from scipy import ndimage
from typing import Dict, Sequence, Optional, Callable

from pathlib import Path
import nibabel
import warnings

warnings.filterwarnings(action='ignore')

class dataset_0127(Dataset):

    def __init__(self, root_dir, path, mode):
        '''
        data loader
            path: path dict list from csv file
            mode: default 'test', chioce = ['test', 'train', 'valid']
            list: output
        '''
        self.path = path
        self.mode = str(mode)
        #self.list = []
        self.root_dir = root_dir

        if mode == 'test':
            self.create_test_data()
        # if mode == 'train':
        #     self.create_train_data()

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        #return self.list[idx]
        if self.mode == 'train':
            ith_info = self.path[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            turelabel_name = os.path.join(self.root_dir, ith_info[1])
            prelabel_name = os.path.join(self.root_dir, ith_info[2])

            if not os.path.isfile(img_name):
                print(img_name)

            assert os.path.isfile(img_name)
            assert os.path.isfile(turelabel_name)
            assert os.path.isfile(prelabel_name)

            img_array, mask_array, orig_affine, affine = load_dataset_tr(img_name, turelabel_name, prelabel_name)
            assert img_array is not None
            assert mask_array is not None

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            #mask_array = self.__nii2tensorarray__(mask_array)

            # assert img_array.shape == mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(
            #     img_array.shape, mask_array.shape)
            return img_array, mask_array, img_name, orig_affine, affine


    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype(np.float)

        return new_data

    def create_train_data(self):
        '''
                img: image array
                affine: low resolution affine
                affine_org: original affine
                shape_org: original image array shape
        '''
        volumes = []
        affines = []
        affine_origs = []
        shapes = []
        img_paths = self.path['img']
        label_paths = self.path['label']
        for i in range(len(img_paths)):
            img, affine, affine_org, shape_org = load_dataset_tr(img_paths[i], self.mode)
            label, affine, affine_org, shape_org = load_dataset_tr(label_paths[i], self.mode)
            self.list.append(tuple((img, label)))


        # img_paths = self.path['img']
        # for file in img_paths:
        #     img, affine, affine_org, shape_org = load_dataset_ts(file, self.mode)
        #     self.list.append(tuple((img, affine, affine_org, shape_org)))
        #
        # label_paths = self.path['label']
        # for file in label_paths:
        #     img, affine, affine_org, shape_org = load_dataset_ts(file, self.mode)
        #     self.labellist.append(tuple((img, affine, affine_org, shape_org)))

        total = len(img)

        # self.list.append(tuple((img, affine, affine_org, shape_org)))
        #total = len(img)

        print("end", flush=True)

    def create_test_data(self):
        '''
        img: image array
        affine: low resolution affine
        affine_org: original affine
        shape_org: original image array shape
        '''
        img, affine, affine_org, shape_org = load_dataset_tr(self.path, self.mode)
        total = len(img)

        self.list.append(tuple((img, affine, affine_org, shape_org)))


        print("end", flush=True)


def load_data_tr(img_path, label_path,  normalize=True):
    ## nifti file load
    #volume_nifty = nb.load(file_path['img'])
    volume_nifty = nb.load(img_path)
    mask_nifty = nb.load(label_path)

    ### nifti file info
    volume_orig_shape = volume_nifty.get_fdata().shape
    volume_orig_affine = volume_nifty.affine

    ### down sampling
    volume_nifty_re = resample_img(
        img=volume_nifty,
        target_affine=np.diag([1.5, 1.5, 3]),
        interpolation='continuous'
    )
    mask_nifty_re= resample_img(
        img=mask_nifty,
        target_affine=np.diag([1.5, 1.5, 3]),
        interpolation='continuous'
    )
    affine = volume_nifty.affine
    volume = volume_nifty_re.get_fdata()
    label = mask_nifty_re.get_fdata()

    # volume, label = drop_invalid_range(volume, label)
    #######裁剪只留左侧
    a, b, c = np.shape(volume)
    # volume = volume[a - int((a * 5) / 8):, :, :]
    # label = label[a - int((a * 5) / 8):, :, :]

    # resize data
    # volume = resize_data(volume, 128, 128, 64)
    # label = resize_data(label,  128, 128, 64)
    # 归一化
    # if normalize:
    #     volume = itensity_normalize_one_volume(volume)
    if normalize:
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    return volume, label, volume_orig_affine, affine

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def drop_invalid_range(volume, label):
    """
    Cut off the invalid area
    """
    zero_value = volume[0, 0, 0]
    non_zeros_idx = np.where(volume != zero_value)

    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

    return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]

def resize_data(data, input_D, input_H, input_W):
    """
    Resize the data to the input size
    """
    [depth, height, width] = data.shape
    scale = [input_D * 1.0 / depth, input_H * 1.0 / height, input_W * 1.0 / width]
    data = ndimage.interpolation.zoom(data, scale, order=0)

    return data

def load_dataset_tr(img_path, turelabelpath, prelabelpath, normalize=True):

    volume, label, orig_affine, affine = load_data_tr(img_path, turelabelpath,prelabelpath, normalize)

    return volume, label, orig_affine, affine


###################################################################################
'''
Article{miscnn,
  title={MIScnn: A Framework for Medical Image Segmentation with Convolutional Neural Networks and Deep Learning},
  author={Dominik Müller and Frank Kramer},
  year={2019},
  eprint={1910.09308},
  archivePrefix={arXiv},
  primaryClass={eess.IV}
}

https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/utils/patch_operations.py
'''


def slice_3Dmatrix(array, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((len(array) - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((len(array[0]) - overlap[1]) /
                            float(window[1] - overlap[1])))
    steps_z = int(math.ceil((len(array[0][0]) - overlap[2]) /
                            float(window[2] - overlap[2])))
    # print(len(array))
    # print(len(array[0]))
    # print(len(array[0][0]))
    # Iterate over it x,y,z
    patches = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Define window edges
                x_start = x * window[0] - x * overlap[0]
                x_end = x_start + window[0]
                y_start = y * window[1] - y * overlap[1]
                y_end = y_start + window[1]
                z_start = z * window[2] - z * overlap[2]
                z_end = z_start + window[2]
                # Adjust ends
                if (x_end > len(array)):
                    # Create an overlapping patch for the last images / edges
                    # to ensure the fixed patch/window sizes
                    x_start = len(array) - window[0]
                    x_end = len(array)
                    # Fix for MRIs which are smaller than patch size
                    if x_start < 0: x_start = 0
                if (y_end > len(array[0])):
                    y_start = len(array[0]) - window[1]
                    y_end = len(array[0])
                    # Fix for MRIs which are smaller than patch size
                    if y_start < 0: y_start = 0
                if (z_end > len(array[0][0])):
                    z_start = len(array[0][0]) - window[2]
                    z_end = len(array[0][0])
                    # Fix for MRIs which are smaller than patch size
                    if z_start < 0: z_start = 0
                # Cut window
                window_cut = array[x_start:x_end, y_start:y_end, z_start:z_end]
                # Add to result list
                patches.append(window_cut)

    return patches


def concat_3Dmatrices(patches, image_size, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((image_size[0] - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((image_size[1] - overlap[1]) /
                            float(window[1] - overlap[1])))
    steps_z = int(math.ceil((image_size[2] - overlap[2]) /
                            float(window[2] - overlap[2])))

    # Iterate over it x,y,z
    matrix_x = None
    matrix_y = None
    matrix_z = None
    pointer = 0
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Calculate pointer from 3D steps to 1D list of patches
                pointer = z + y * steps_z + x * steps_y * steps_z
                # Connect current patch to temporary Matrix Z
                if z == 0:
                    matrix_z = patches[pointer]
                else:
                    matrix_p = patches[pointer]
                    # Handle z-axis overlap
                    slice_overlap = calculate_overlap(z, steps_z, overlap,
                                                      image_size, window, 2)
                    matrix_z, matrix_p = handle_overlap(matrix_z, matrix_p,
                                                        slice_overlap,
                                                        axis=2)
                    matrix_z = np.concatenate((matrix_z, matrix_p),
                                              axis=2)
            # Connect current tmp Matrix Z to tmp Matrix Y
            if y == 0:
                matrix_y = matrix_z
            else:
                # Handle y-axis overlap
                slice_overlap = calculate_overlap(y, steps_y, overlap,
                                                  image_size, window, 1)
                matrix_y, matrix_z = handle_overlap(matrix_y, matrix_z,
                                                    slice_overlap,
                                                    axis=1)
                matrix_y = np.concatenate((matrix_y, matrix_z), axis=1)
        # Connect current tmp Matrix Y to final Matrix X
        if x == 0:
            matrix_x = matrix_y
        else:
            # Handle x-axis overlap
            slice_overlap = calculate_overlap(x, steps_x, overlap,
                                              image_size, window, 0)
            matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                slice_overlap,
                                                axis=0)
            matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
    # Return final combined matrix
    return (matrix_x)


# -----------------------------------------------------#
#          Subroutines for the Concatenation          #
# -----------------------------------------------------#
# Calculate the overlap of the current matrix slice
def calculate_overlap(pointer, steps, overlap, image_size, window, axis):
    # Overlap: IF last axis-layer -> use special overlap size
    if pointer == steps - 1 and not (image_size[axis] - overlap[axis]) \
                                    % (window[axis] - overlap[axis]) == 0:
        current_overlap = window[axis] - \
                          (image_size[axis] - overlap[axis]) % \
                          (window[axis] - overlap[axis])
    # Overlap: ELSE -> use default overlap size
    else:
        current_overlap = overlap[axis]
    # Return overlap
    return current_overlap


# Handle the overlap of two overlapping matrices
def handle_overlap(matrixA, matrixB, overlap, axis):
    # Access overllaping slice from matrix A
    idxA = [slice(None)] * matrixA.ndim
    matrixA_shape = matrixA.shape
    idxA[axis] = range(matrixA_shape[axis] - overlap, matrixA_shape[axis])
    sliceA = matrixA[tuple(idxA)]
    # Access overllaping slice from matrix B
    idxB = [slice(None)] * matrixB.ndim
    idxB[axis] = range(0, overlap)
    sliceB = matrixB[tuple(idxB)]
    # Calculate Average prediction values between the two matrices
    # and save them in matrix A
    matrixA[tuple(idxA)] = np.mean(np.array([sliceA, sliceB]), axis=0)
    # Remove overlap from matrix B
    matrixB = np.delete(matrixB, [range(0, overlap)], axis=axis)
    # Return processed matrices
    return matrixA, matrixB




