# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:35:51 2023

@author: Maksim
"""

# import cv2
import os
import copy
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage import io


def add_mirroring_img_down(im, size):
    buffer = np.flip(im, axis=0)
    if len(im.shape) == 3:
        im = np.concatenate((im, buffer[:size, :, :]), axis=0)
    else:
        im = np.concatenate((im, buffer[:size, :]), axis=0)
    return im


def add_mirroring_img_right(im, size):
    buffer = np.flip(im, axis=1)
    if len(im.shape) == 3:
        im = np.concatenate((im, buffer[:, :size, :]), axis=1)
    else:
        im = np.concatenate((im, buffer[:, :size]), axis=1)
    return im


def image_slicing(im, size):
    
    num_str = int(im.shape[0]/size)
    num_col = int(im.shape[1]/size)
    
    del_str = im.shape[0] - (size * num_str)
    del_col = im.shape[1] - (size * num_col)
    res_im = copy.deepcopy(im)

    if del_str != 0:
        res_im = add_mirroring_img_down(im, (size-del_str))

    if del_col != 0:    
        res_im = add_mirroring_img_right(res_im, (size-del_col))
    
    num_str = int(res_im.shape[0]/size)
    num_col = int(res_im.shape[1]/size)
    
    sli_img = list()
    for i in range(num_str):
        for j in range(num_col):
            if len(im.shape) == 3:
                sli_img.append(res_im[int(size * i):int(size + (size*i)),
                                      int(size * j):int(size + (size*j)),
                                      :3])
            else:
                sli_img.append(res_im[int(size * i):int(size + (size * i)),
                               int(size * j):int(size + (size * j))])
    return(sli_img)


def save_sli_img(sli_img, name, output_path):

    for i in range(len(sli_img)):        
        output_name = f"{name}_cut_{i+1}.png"

        io.imsave(output_path[0] + '\\' + output_name, sli_img[i],  check_contrast=False)


if __name__ == '__main__':

    size = 512  #  size of output image
    images_path = glob(os.path.join('C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData\\RatBrain_stainnorm', '*.png'))  # path of input images for cut
    output_path = glob(os.path.join('C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData\\PP_train_images_all'))  # output path

    for i in tqdm(range(len(images_path)), total=len(images_path)):
        print(images_path[i])
        im = io.imread(fname=images_path[i])
        sli_img = image_slicing(im, size)

        name = images_path[i]
        name = name.split("\\")[-1].split(".")[0]
        save_sli_img(sli_img, name, output_path)
