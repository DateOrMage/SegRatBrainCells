from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import os
import torch
from torch_unet_universal import UNET
from torchvision import transforms
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy
from torch_metrics import DiceMetric, DiceBCELoss


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def pred_one_file(image_path, model_path):
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    model = load_checkpoint(model_path)
    model = model.to('cpu')
    try:
        model = model.module
    except AttributeError:
        model = model

    image = io.imread(image_path)
    image = img_as_ubyte(image)  # if float img type
    image = trans(image)
    image = torch.reshape(image, (1, 3, 512, 512))
    image = image.to('cpu')
    pred = model(image)
    pred = pred.cpu().numpy()[0][0]
    pred = (pred > 0.5) * 1
    plt.imshow(pred, cmap='gray')


def predict(image_path, mask_path, model_path, out_path, is_save_double=False, is_metric=False):
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    model = load_checkpoint(model_path)
    model = model.to('cpu')
    try:
        model = model.module
    except AttributeError:
        model = model

    total_dice = 0
    total_precision = 0
    total_recall = 0

    image_list = os.listdir(image_path)  # [1:7]
    mask_list = os.listdir(mask_path)  # [1:7]

    for i, image_name in enumerate(image_list):
        print(f'{i}/{len(image_list)}')
        image_name = image_name.split(sep='.')[0] + '.png'
        image = io.imread(os.path.join(image_path, image_list[i]))
        image = trans(image)
        image = torch.reshape(image, (1, 3, 512, 512))
        image = image.to('cpu')
        pred = model(image)
        pred = (pred > 0.5) * 1

        mask = io.imread(os.path.join(mask_path, mask_list[i]))
        mask = trans(mask)
        mask = torch.reshape(mask, (1, 1, 512, 512))
        mask = mask.to('cpu')
        mask = (mask > 0.5) * 1

        if is_metric:
            dm = DiceMetric()
            metric = dm.forward(pred, mask)
            total_dice += metric.item()
            print(f'{image_name}:\n{metric} - Dice coef')
            bp = BinaryPrecision()
            metric = bp(pred, mask)
            total_precision += metric.item()
            print(f'{metric} - Precision')
            br = BinaryRecall()
            metric = br(pred, mask)
            total_recall += metric.item()
            print(f'{metric} - Recall')

        pred = pred.cpu().numpy()[0][0]

        if is_save_double:
            mask = mask.cpu().numpy()[0][0]
            figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax[0].imshow(mask, cmap='gray')
            ax[1].imshow(pred, cmap='gray')
            ax[0].set_title('Original mask')
            ax[1].set_title('Predicted mask')
            figure.tight_layout()
            figure.savefig(os.path.join(out_path, image_name), dpi=500)
            plt.close()
        else:
            io.imsave(out_path + '\\' + image_name, pred, check_contrast=False)
            continue
    if is_metric:
        avg_metric_dice = total_dice / len(image_list)
        avg_metric_precision = total_precision / len(image_list)
        avg_metric_recall = total_recall / len(image_list)
        print(f'Total average metrics on {len(image_list)} images: \n{avg_metric_dice} - Dice coef'
              f' \n{avg_metric_precision} - Precision \n{avg_metric_recall} - Recall')


if __name__ == '__main__':

    predict(
        image_path="C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData\\PP_test_images",
        mask_path="C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData\\PP_test_masks",
        model_path='iunet4_32_BN_DO_Adam_stain_reinhard.pth',
        out_path="C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData\\output_all_4_32_reinhard",
        is_save_double=True,
        is_metric=True)

    # pred_one_file("C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\Cell images ver4\\PP_test_images\\Original image (1)_cut_1_left_right.png",
                  # 'unet3_32_BN_DO_Adam_grad_0_001_torch_all.pth')

