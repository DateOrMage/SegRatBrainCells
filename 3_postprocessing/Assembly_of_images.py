# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:13:51 2023

@author: Maksim
"""
import cv2
import os
import numpy as np
from glob import glob

images_path = glob(os.path.join('Input', 'ImgPred', '*.png'))  # input images path
output_path = 'Output'  # output path

width = 3  # quantity image by width
height = 3  # quantity image by height

resolution_width = 1376  # output width
resolution_height = 1032  # output height

number_lines = int(len(images_path)/width)

    
counter = 0
list_with_lines = list()

for i in range(number_lines):
    lines = cv2.imread(images_path[counter], cv2.IMREAD_COLOR)
    counter = counter + 1
    for j in range(1, width):
        lines = np.concatenate((lines, cv2.imread(images_path[counter], cv2.IMREAD_COLOR)), axis=1)
        counter = counter + 1
    
    list_with_lines.append(lines)
    

counter = 0
number_output_images = int(number_lines/height)
list_with_output_images = list()

for i in range(number_output_images):
    image = list_with_lines[counter]
    counter = counter + 1
    for j in range(1, height):
        image = np.concatenate((image, list_with_lines[counter]), axis=0)
        counter = counter + 1
    
    list_with_output_images.append(image)
    
for i in range(len(list_with_output_images)):
    name = images_path[int(i*width*height)]
    name = name.split("\\")[-1].split(".")[0].split("_")[0]
    tpm_name = f"{name} result.png"
    output_image = list_with_output_images[i][:resolution_height, :resolution_width, :]
    cv2.imwrite(os.path.join(output_path, tpm_name), output_image)
    
    