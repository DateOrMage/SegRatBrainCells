# !pip install staintools
# !pip install spams # use python notebook, if u cannot install "spams" on ur machine

import staintools
import os
from skimage import io, img_as_ubyte


def run_vahadane(path_in, path_out, target_name=None):
    images_list = os.listdir(path_in)
    if target_name is None:
        target_name = 'Original image (135).png'

    img_target = io.imread(os.path.join(path_in, target_name))

    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(img_target)

    for file_name in images_list:
        img_path = os.path.join(path_in, file_name)
        image = staintools.read_image(img_path)
        trans_img = normalizer.transform(image)

        res = file_name.split(' ', 1)
        out_name = 'Stannorm_vaha ' + res[1]
        print(out_name)
        img_path_out = os.path.join(path_out, out_name)

        io.imsave(img_path_out, img_as_ubyte(trans_img), check_contrast=False)


if __name__ == '__main__':
    run_vahadane(path_in='path\\to\\original\\image',
                 path_out='path\\to\\save\\dir')


