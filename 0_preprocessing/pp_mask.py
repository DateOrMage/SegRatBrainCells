from skimage import io, img_as_ubyte
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def open_image(img_path):
    image = io.imread(fname=img_path)
    plt.imshow(image)
    return image


def draw(pic):
    fig = plt.figure()
    plt.imshow(pic[:, :, 0], cmap='Reds')
    fig1 = plt.figure()
    plt.imshow(pic[:, :, 1], cmap='Greens')
    fig2 = plt.figure()
    plt.imshow(pic[:, :, 2], cmap='Blues')
    fig3, ax = plt.subplots(1, 1)
    ax.hist(pic[:, :, 0].ravel(), bins=32, range=[0, 256], color='red')
    ax.set_xlim(0, 256)
    fig4, ax1 = plt.subplots(1, 1)
    ax1.hist(pic[:, :, 1].ravel(), bins=32, range=[0, 256], color='green')
    ax1.set_xlim(0, 256)
    fig5, ax2 = plt.subplots(1, 1)
    ax2.hist(pic[:, :, 2].ravel(), bins=32, range=[0, 256], color='blue')
    ax2.set_xlim(0, 256)


def pp_mask(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] > 250:
                img[i, j, 0] = 255
                img[i, j, 1] = 255
                img[i, j, 2] = 255
            else:
                img[i, j, 0] = 0
                img[i, j, 1] = 0
                img[i, j, 2] = 0
    return rgb2gray(img)


def run_pp_mask(path_in, path_out):
    import os
    files = os.listdir(path_in)
    for file_name in files:
        print(file_name)
        mask = io.imread(fname=path_in+'\\'+file_name)
        if mask.shape[-1] == 4:
            mask = mask[:, :, :3]
        mask = pp_mask(mask)
        file_name_png = file_name.split('.')[0] + '.png'
        io.imsave(path_out+'\\'+file_name_png, img_as_ubyte(mask), check_contrast=False)


if __name__ == '__main__':
    run_pp_mask(path_in='C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\Cell images ver4\\Labeled images',
                path_out='C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\Cell images ver4\\Mask images')








