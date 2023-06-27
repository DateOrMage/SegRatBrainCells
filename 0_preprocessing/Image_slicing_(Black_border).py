# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:23:23 2023

@author: Maksim
"""

# import cv2
import os
import copy
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage import io


def add_black_line_down(im, size):
    buffer = np.zeros((size, im.shape[1], 3)).astype(np.uint8)
    im = np.concatenate((im, buffer), axis=0)
    return im


def add_black_line_right(im, size):
    buffer = np.zeros((im.shape[0], size, 3)).astype(np.uint8)
    im = np.concatenate((im, buffer), axis=1)
    return im


def image_slicing(im, size):

    print(im)
    num_str = int(im.shape[0]/size)
    num_col = int(im.shape[1]/size)
    
    del_str = im.shape[0] - (size * num_str)
    del_col = im.shape[1] - (size * num_col)
    res_im = copy.deepcopy(im)

    if del_str != 0:
        res_im = add_black_line_down(im, (size-del_str))

    if del_col != 0:    
        res_im = add_black_line_right(res_im, (size-del_col))
    
    num_str = int(res_im.shape[0]/size)
    num_col = int(res_im.shape[1]/size)
    
    sli_img = list()
    for i in range(num_str):
        for j in range(num_col):
            sli_img.append(res_im[int(size * i):int(size + (size*i)),
                                  int(size * j):int(size + (size*j)),
                                  :3])
    return(sli_img)


def save_sli_img(sli_img, name, output_path):

    for i in range(len(sli_img)):        
        output_name = f"{name}_cut_{i+1}.png"
        io.imsave(output_path[0] + '\\' + output_name, sli_img[i],  check_contrast=False)


if __name__ == '__main__':

    size = 512  # size of output image
    images_path = glob(os.path.join('C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\Cell images ver4\\ImgTesting\\ImgOriginal', '*.png'))  # path of input images for cut
    output_path = glob(os.path.join('C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\Cell images ver4\\ImgTesting\\ImgCut'))  # output path

    for i in tqdm(range(len(images_path)), total=len(images_path)):

        print(images_path[i])
        im = io.imread(fname=images_path[i])
        sli_img = image_slicing(im, size)

        name = images_path[i]
        name = name.split("\\")[-1].split(".")[0]
        save_sli_img(sli_img, name, output_path)

