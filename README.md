# Segmantation of rat brain cells by U-Net

Python scripts for training and evalution semantic segmentation of rat brain cells.

## Files description
0_preprocessing - python scripts for prepared data before training

1_training - scripts for training model.

2_predict - scripts for prediction masks from images.

3_postprocessing - scripts for processing masks after prediction.

Labeled_images - images withmarked alive neurons.

Original images - unprocessed image with stain trypan blue.

test_images - images for testing without labeled masks.

trained_models - trained models only U-Net 3-16 and 3-32 because file more then 25 mb cannot upload to github.

requirements.txt - file with python libs, witch have to install

Install python libs with pip:
```bash
pip install requirements.txt
```

## Prepeared data before training
First of all we have to prepeared mask from labeled images. There is python script "pp_mask.py" in directory "0_preprocessing". Run the script with changed paths to getting masks:
```bash
57 if __name__ == '__main__':
58    run_pp_mask(path_in='path\\to\\directory\\with\\labeled images',
59                path_out='path\\to\\directory\\where\\save\\mask images')
```
Next step is stain-normilization original images. It is optional performance. Choise reinhard or vahadane or nothing.

If reinhard is choisen (stain_norm_reinhart.py):
```bash
84 if __name__ == '__main__':
85    run_reinhart(path_in='path\\to\\original\\image',
86                 path_out='path\\to\\save\\dir')
```
If vahadane is chosen (stain_norm_vahadane.py):
```bash
32 if __name__ == '__main__':
33    run_vahadane(path_in='path\\to\\original\\image',
34                 path_out='path\\to\\save\\dir')
```
Next step is slicing images by 512x512 (default) fragments. Choise black border or mirror border (prefer for training). Notes that performance have to use on masks too!!!
For black border (Image_slicing_(Black_border).py):
```bash
64 if __name__ == '__main__':
65
66    size = 512  # size of output image
67    images_path = glob(os.path.join('path\\to\\input\\images\\for\\cut', '*.png'))
68    output_path = glob(os.path.join('path\\to\\save\\fragments'))
```
Same for mirror border (Image_slicing_(Mirror_border).py):
```bash
74 if __name__ == '__main__':
75
76    size = 512  # size of output image
77    images_path = glob(os.path.join('path\\to\\input\\images\\for\\cut', '*.png'))
78    output_path = glob(os.path.join('path\\to\\save\\fragments'))
```
Last step is augmentation, used for images and masks. Run the augmentation.py:
```bash
35 if __name__ == '__main__':
36    data_path = 'path\\to\\images\\for\\augmentation'
37    mask_path = 'path\\to\\masks\\for\\augmentation'
38
39    run_aug(data_path)  # augmentation's images will save to same directory
```

After all steps we have to have prepared images for training and them masks.

## Training
First of all have to check "torch_config.py" for change training's parameters.
```bash
import torch
import os

DATASET_PATH = 'C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData'  # path to directory with prepared images and masks

IMAGE_PATH = os.path.join(DATASET_PATH, 'PP_train_images_all')  # add path to prepared images
MASK_PATH = os.path.join(DATASET_PATH, 'PP_train_masks_all')  # add path to prepared masks

TEST_SPLIT = 0.20  # persent of all data to validation samples 

BATCH_SIZE = 32  # number of images in 1 batch
NUM_EPOCHS = 300  # max number of epochs
EPOCHS_NO_IMPROVE = 10  # number of epochs before early stopping

SEED_COUNTER = 1  # number of starting with random seed (1 is prefer)

CNN_DEPTH = [4]  # depth of U-Net (number of encoder/decoder blocks)
FIRST_CHANNELS = [32]  # number of conv_kernel in first encoder block
BATCH_NORM = [True]  # add BatchNormalization2d layers or not
DROP_OUT = [True]  # add Dropout(p=0.3) layers or not

OPTIMIZER = 'Adam'  # SGD, RMSprop (Adam is prefer)
AMSGRAD = [False]  # [False, True]  # (False is prefer) 
LR = [0.001]  # learning rate
MOMENT = [(0.9, 0.999)]  # betas
WEIGHT_DECAY = [0]  # L2 penalty (0 is prefer)

IS_SAVE = True  # save model after training or not
BASE_OUTPUT = 'output'  # output path
MODEL_NAME = 'unet_Adam_all_data.pth'  # basic name of model
PATH_OUTPUT = os.path.join(DATASET_PATH, BASE_OUTPUT)  # path to save history of train/valid loss function
MODEL_OUTPUT = os.path.join(DATASET_PATH, BASE_OUTPUT, MODEL_NAME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # choise gpu if cuda there is, else cpu
PIN_MEMORY = True if DEVICE == "cuda" else False
```
For start training run "torch_training.py":
```bash
297 if __name__ == '__main__':
298     hyper_best_dict, hyper_total_dict, history_train = grid_search_adam()
```

## Prediction
For evalution trained model have to use "torch_predict.py". For succesfully prediction must have prepeared images and trained model. If you want to get compare with true masks, you have to have prepeared masks.
```bash
111 if __name__ == '__main__':
112
113     predict(
114         image_path="path\\to\\prepeared\\images",
115         mask_path="path\\to\\prepeared\\masks",
116         model_path='iunet4_32_BN_DO_Adam_stain_reinhard.pth', # path to model
117         out_path="C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData\\output_all_4_32_reinhard",
118         is_save_double=True, # save double image (original mask + predicted) or not
119         is_metric=True)  # print metric or not
120     # you can predict only 1 images without save and comparing
121     # pred_one_file("C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\Cell images ver4\\PP_test_images\\Original image (1)_cut_1_left_right.png",
122                   # 'unet3_32_BN_DO_Adam_grad_0_001_torch_all.pth')
```
















