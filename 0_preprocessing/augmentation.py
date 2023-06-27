import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rotate
from skimage import img_as_ubyte


def augmentation_img(mode, image):
    if mode == 'rotate+90':
        image_aug = rotate(image, angle=90)
    elif mode == 'rotate-90':
        image_aug = rotate(image, angle=-90)
    elif mode == 'rotate+180':
        image_aug = rotate(image, angle=180)
    elif mode == 'left_right':
        image_aug = np.fliplr(image)
    elif mode == 'up_down':
        image_aug = np.flipud(image)
    else:
        raise Exception(f'Incorrect mode: {mode}')
    return image_aug


def run_aug(path):
    files_list = os.listdir(path)
    for file_name in files_list:
        image = imread(path + '\\' + file_name)
        for mode in ['rotate+90', 'rotate-90', 'rotate+180', 'left_right', 'up_down']:
            image_aug = augmentation_img(mode=mode, image=image)
            file_name_save = file_name.split('.')[0] + f'_{mode}' + '.' + file_name.split('.')[1]
            print(file_name_save)
            imsave(path + '\\' + file_name_save, img_as_ubyte(image_aug), check_contrast=False)


if __name__ == '__main__':
    data_path = 'C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData\\PP_train_images_all'  # path to images for augmentation
    mask_path = 'C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\Cell images ver4\\PP_train_masks'  # path to masks for augmentation

    run_aug(data_path)  # run augmentation


