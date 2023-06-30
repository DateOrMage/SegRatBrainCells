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

## Order for training models
