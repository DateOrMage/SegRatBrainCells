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

## Order for training models
First of all have to check "torch_config.py" for change training's parameters.
```bash
import torch
import os

DATASET_PATH = 'C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData'

IMAGE_PATH = os.path.join(DATASET_PATH, 'PP_train_images_all')
MASK_PATH = os.path.join(DATASET_PATH, 'PP_train_masks_all')

TEST_SPLIT = 0.20

BATCH_SIZE = 32
NUM_EPOCHS = 300  # 300
EPOCHS_NO_IMPROVE = 10

SEED_COUNTER = 1

CNN_DEPTH = [4]  # [3, 4, 5, 6, 7, 8, 9]
FIRST_CHANNELS = [32]  # [32, 64]
BATCH_NORM = [True]
DROP_OUT = [True]

OPTIMIZER = 'Adam'  # SGD, RMSprop
AMSGRAD = [False]  # [False, True]
LR = [0.001]  # [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
MOMENT = [(0.9, 0.999)]  # [(0.9, 0.99), (0.9, 0.999), (0.9, 0.9999)]
WEIGHT_DECAY = [0]  # [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

IS_SAVE = True
BASE_OUTPUT = 'output'
MODEL_NAME = 'unet_Adam_all_data.pth'
PATH_OUTPUT = os.path.join(DATASET_PATH, BASE_OUTPUT)
MODEL_OUTPUT = os.path.join(DATASET_PATH, BASE_OUTPUT, MODEL_NAME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
```
















