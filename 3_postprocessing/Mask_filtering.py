# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:11:28 2023

@author: Maksim
"""

import csv
import copy
import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import morphology
from skimage.measure import label, regionprops, regionprops_table
from scipy.io import savemat 

images_path = glob(os.path.join('Input', '*.png'))  # masks input path
output_path = 'Output'  # output path

remove_area = 150  # drop areas with square less than 150
remove_eccentricity = 0.9  # drop areas with eccentricity coef less than 0.9

for i in range(len(images_path)):
    
    name = images_path[i]
    name = name.split("\\")[-1].split(".")[0]
    tpm_name = f"{name} result.png"
    
    im = cv2.imread(images_path[i], cv2.IMREAD_COLOR)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    bw = im_gray > 250
    bw = morphology.remove_small_objects(bw, min_size=remove_area)
    
    label_bw = label(bw)
    regions = regionprops(label_bw)
    
    centroid_list = list()
    equivalent_diametr_list = list()
    eccentricity_list = list()
    
    for j in range(len(regions)):
        if regions[j].eccentricity < remove_eccentricity:
            
            cv2.circle(im, (int(regions[j].centroid[1]), int(regions[j].centroid[0])),
                       int(regions[j].equivalent_diameter_area), (6, 6, 255), (2))
            centroid_list.append(regions[j].centroid)
            equivalent_diametr_list.append(regions[j].equivalent_diameter_area)
            #  eccentricity_list.append(regions[j].eccentricity)
    
    cv2.imwrite(os.path.join(output_path, tpm_name), im)