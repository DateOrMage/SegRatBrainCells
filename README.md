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
Next step is stain-normilization original images. It is optional performance. Choise reinhart or vahadane or nothing.

If reinhart is choisen:
```bash
84 if __name__ == '__main__':
85    run_reinhart(path_in='path\\to\\original\\image',
86                 path_out='path\\to\\save\\dir')
```
If vahadane is chosen:
```bash
32 if __name__ == '__main__':
33    run_vahadane(path_in='path\\to\\original\\image',
34                 path_out='path\\to\\save\\dir')
```
















## Order for training models
