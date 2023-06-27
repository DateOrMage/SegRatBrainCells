import numpy as np
import os
from skimage import io, color, img_as_ubyte
from skimage.io import imsave, imread


def matrix_shift(target):
    [x1, y1, z1] = np.shape(target)

    target = np.float64(target)/255
    target = color.rgb2lab(target)

    tar = np.zeros((x1*y1, 3))
    for i in np.arange(0, 3):
        tar[:, i] = target[:, :, i].flatten('F')

    m_tar = np.zeros((1, 3))
    std_tar = np.zeros((1, 3))
    for i in np.arange(0, 3):
        m_tar[0, i] = np.mean(tar[:, i])
        std_tar[0, i] = np.std(tar[:, i])

    target_mat = np.array([m_tar, std_tar])
    target_mat = np.squeeze(target_mat)

    return target_mat


def normalize_to_target(target_mat, source):

    import numpy as np
    from skimage import color

    [x, y, z] = np.shape(source)

    source = np.float64(source)/255
    source = color.rgb2lab(source)

    src = np.zeros((x*y, 3))
    for i in np.arange(0, 3):
        src[:, i] = source[:, :, i].flatten('F')

    m_src = np.zeros((1, 3))
    std_src = np.zeros((1, 3))
    for i in np.arange(0, 3):
        m_src[0, i] = np.mean(src[:, i])
        std_src[0, i] = np.std(src[:, i])

    result = np.zeros((x, y, 3))
    for k in np.arange(0, 3):
        for i in np.arange(0, x):
            for j in np.arange(0, y):
                result[i, j, k] = ((source[i, j, k] - m_src[0, k]) * (target_mat[1, k] / std_src[0, k])) + target_mat[0, k]

    result = color.lab2rgb(result)
    rgb_result = result*255
    rgb_result = np.uint8(rgb_result)

    return rgb_result


def run_reinhart(path_in, path_out, target_name=None):
    images_list = os.listdir(path_in)
    if target_name is None:
        target_name = 'Original image (135).png'

    img_target = io.imread(os.path.join(path_in, target_name))

    target_matrix = matrix_shift(target=img_target)

    for file_name in images_list:
        img_path = os.path.join(path_in, file_name)
        image = imread(img_path)
        trans_img = normalize_to_target(target_matrix, source=image)

        res = file_name.split(' ', 1)
        out_name = 'Stannorm_rein ' + res[1]
        print(out_name)
        img_path_out = os.path.join(path_out, out_name)

        imsave(img_path_out, img_as_ubyte(trans_img), check_contrast=False)


if __name__ == '__main__':
    run_reinhart(path_in='path\\to\\original\\image',
                 path_out='path\\to\\save\\dir')
