# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:55:32 2023

@author: Maksim
"""

import cv2
import os
import numpy as np
from glob import glob
from skimage import morphology
from skimage.measure import label, regionprops
from tqdm import tqdm


def add_black_line_down(im, size):
    buffer = np.zeros((size, im.shape[1], 3)).astype(np.uint8)
    im = np.concatenate((im, buffer), axis=0)
    return im


images_path = glob(os.path.join('Input', '*.png'))  # input path of images
output_path = glob(os.path.join('Output'))  # output path

for i in tqdm(range(len(images_path)), total=len(images_path)):

    im = cv2.imread(images_path[i], cv2.IMREAD_COLOR)
    name = images_path[i]
    name = name.split("\\")[-1].split(".")[0]
    output_name = f"{name}_with_number_of_cells.png"
    
    bw = im[:, :, 2] > 254
    bw = morphology.remove_small_objects(bw, min_size = 100)
    bw = ~bw
    bw = morphology.remove_small_objects(bw, min_size = 15)
    bw = ~bw
    bw_2 = bw.astype(np.uint8)
    bw_2 = bw_2 * 255
    
    """ Part of code witch counting cells on image """

    label_bw = label(bw)
    regions = regionprops(label_bw)        
    coordinates = np.empty([np.size(regions, 0), 3], dtype=np.float32)
         
    for k in range(len(regions)):
        coordinates[k, 0:2] = np.asarray(regions[k].centroid)
        coordinates[k, 2] = regions[k].equivalent_diameter_area
 
    num_cells = np.size(coordinates, 0)  # counter

    # ----------------------------------------------
    
    im = add_black_line_down(im, 40)
    im = cv2.putText(im, f'Number of cells: {int(num_cells)}', (10, np.size(im, 0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_path[0], output_name), im)
