import json
import os.path
import torch
import numpy as np
import torch_config
from torch_unet_model import Unet
from torch_unet_universal import UNET
from torch_image_to_tesor import get_data
from torch_metrics import DiceMetric, DiceBCELoss
from torch.optim import Adam, SGD, RMSprop
from torch.nn import BCEWithLogitsLoss, DataParallel
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage import io, transform, img_as_ubyte
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy
import math
import pickle as pkl


def grid_search_adam():
    total_start = time.time()
    best_model = {'loss': 10000, 'metric': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'optimizer': 'Adam', 'cnn_depth': None, 'first_channel': None, 'batch_norm': False,
                  'drop_out': False, 'amsgrad': False, 'lr': None, 'moment': None, 'w_decay': None}
    grid_search_list = []
    iterations = torch_config.SEED_COUNTER*len(torch_config.CNN_DEPTH)*len(torch_config.FIRST_CHANNELS)*len(torch_config.BATCH_NORM)*len(torch_config.DROP_OUT)*len(torch_config.AMSGRAD)*len(torch_config.LR)*len(torch_config.MOMENT)*len(torch_config.WEIGHT_DECAY)
    iteration = 0
    for seed in range(torch_config.SEED_COUNTER):
        for cnn_depth in torch_config.CNN_DEPTH:
            for first_channel in torch_config.FIRST_CHANNELS:
                for batch_norm in torch_config.BATCH_NORM:
                    for drop_out in torch_config.DROP_OUT:
                        for amsgrad in torch_config.AMSGRAD:
                            for lr in torch_config.LR:
                                for moment in torch_config.MOMENT:
                                    for w_decay in torch_config.WEIGHT_DECAY:
                                        print(f'Grid search: {iteration+1}/{iterations}')
                                        print(f'Training CNN with params:\ncnn_depth: {cnn_depth}\nfirst_channel:'
                                              f' {first_channel}\nbatch_norm: {batch_norm}\ndrop_out: {drop_out}\namsgrad:'
                                              f' {amsgrad}\nlr: {lr}\nmoment: {moment}\nw_decay: {w_decay}')
                                        unet = UNET(3, first_channel, 1, batch_norm=batch_norm, drop_out=drop_out,
                                                    downhill=cnn_depth, padding=0)
                                        unet = DataParallel(unet)
                                        opt = Adam(unet.parameters(), lr=lr, amsgrad=amsgrad, betas=moment, weight_decay=w_decay)
                                        checkpoint = {'model': unet, 'state_dict': None, 'optimizer': None,
                                                      'loss': 10000, 'metric': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
                                        history, checkpoint = training(unet, opt, checkpoint)
                                        if best_model['loss'] > checkpoint['loss']:
                                            best_model['loss'] = checkpoint['loss']
                                            best_model['metric'] = checkpoint['metric']
                                            best_model['accuracy'] = checkpoint['accuracy']
                                            best_model['precision'] = checkpoint['precision']
                                            best_model['recall'] = checkpoint['recall']
                                            best_model['cnn_depth'] = cnn_depth
                                            best_model['first_channel'] = first_channel
                                            best_model['batch_norm'] = batch_norm
                                            best_model['drop_out'] = drop_out
                                            best_model['amsgrad'] = amsgrad
                                            best_model['lr'] = lr
                                            best_model['moment'] = moment
                                            best_model['w_decay'] = w_decay
                                        name_file_art = f'i-{seed}_art_depth-{cnn_depth}_feats-{first_channel}_BN-{batch_norm}_DO-{drop_out}_LR-{lr}_M-{moment}_{torch_config.MODEL_NAME}'
                                        if torch_config.IS_SAVE:
                                            torch.save(checkpoint, os.path.join(torch_config.PATH_OUTPUT, name_file_art))
                                        plot_loss(history, name_file_art)
                                        print(f'{seed}: {best_model}')
                                        grid_search_dict = dict.fromkeys(['metric', 'accuracy', 'precision', 'recall', 'cnn_depth', 'first_channel', 'batch_norm', 'drop_out', 'amsgrad', 'lr', 'moment', 'w_decay'])
                                        grid_search_dict['metric'] = checkpoint['metric']
                                        grid_search_dict['accuracy'] = checkpoint['accuracy']
                                        grid_search_dict['precision'] = checkpoint['precision']
                                        grid_search_dict['recall'] = checkpoint['recall']
                                        grid_search_dict['cnn_depth'] = cnn_depth
                                        grid_search_dict['first_channel'] = first_channel
                                        grid_search_dict['batch_norm'] = batch_norm
                                        grid_search_dict['drop_out'] = drop_out
                                        grid_search_dict['amsgrad'] = amsgrad
                                        grid_search_dict['lr'] = lr
                                        grid_search_dict['moment'] = moment
                                        grid_search_dict['w_decay'] = w_decay
                                        grid_search_list.append(grid_search_dict)
                                        iteration += 1

    total_stop = time.time()
    print(f'Training stopped, time: {np.round((total_stop - total_start) / 60, 2)} min')

    with open(os.path.join(torch_config.PATH_OUTPUT, 'best_model_adam.json'), 'w', encoding='utf-8') as file:
        json.dump(best_model, file, indent=4)

    with open(os.path.join(torch_config.PATH_OUTPUT, 'grid_search_adam_lr_m.pickle'), 'wb') as file:
        pkl.dump(grid_search_list, file)

    return best_model, grid_search_list, history


def grid_search_rmsprop():
    best_model = {'loss': 100000, 'metric': 0, 'optimizer': 'RMSprop', 'cnn_depth': None, 'first_channel': None, 'batch_norm': False,
                  'drop_out': False, 'lr': None}
    iterations = len(torch_config.CNN_DEPTH) * len(torch_config.FIRST_CHANNELS) * len(torch_config.BATCH_NORM) * len(
        torch_config.DROP_OUT) * len(torch_config.LR)
    iteration = 0
    for cnn_depth in torch_config.CNN_DEPTH:
        for first_channel in torch_config.FIRST_CHANNELS:
            for batch_norm in torch_config.BATCH_NORM:
                for drop_out in torch_config.DROP_OUT:
                    for lr in torch_config.LR:
                        print(f'Grid search: {iteration+1}/{iterations}')
                        print(f'Training CNN with params:\ncnn_depth: {cnn_depth}\nfirst_channel: {first_channel}\nbatch_norm:{batch_norm}\ndrop_out: {drop_out}\nlr: {lr}')
                        unet = UNET(3, first_channel, 1, batch_norm=batch_norm, drop_out=drop_out,
                                    downhill=cnn_depth, padding=0)
                        opt = RMSprop(unet.parameters(), lr=lr, centered=True)
                        checkpoint = {'model': unet, 'state_dict': None, 'optimizer': None,
                                      'loss': 10000, 'metric': 0}
                        history = training(unet, opt, checkpoint)
                        if best_model['loss'] > min(history['valid_loss']):
                            best_model['metric'] = max(history['valid_metric'])
                            best_model['cnn_depth'] = cnn_depth
                            best_model['first_channel'] = first_channel
                            best_model['batch_norm'] = batch_norm
                            best_model['drop_out'] = drop_out
                            best_model['lr'] = lr
                        print(best_model)
                        iteration += 1

    with open('best_model_rmsprop.json', 'w', encoding='utf-8') as file:
        json.dump(best_model, file, indent=4)

    return best_model


def grid_search_sgd():
    best_model = {'metric': 0, 'optimizer': 'SGD', 'cnn_depth': None, 'first_channel': None, 'batch_norm': False,
                  'drop_out': False, 'lr': None}
    iterations = len(torch_config.CNN_DEPTH) * len(torch_config.FIRST_CHANNELS) * len(torch_config.BATCH_NORM) * len(
        torch_config.DROP_OUT) * len(torch_config.LR)
    iteration = 0
    for cnn_depth in torch_config.CNN_DEPTH:
        for first_channel in torch_config.FIRST_CHANNELS:
            for batch_norm in torch_config.BATCH_NORM:
                for drop_out in torch_config.DROP_OUT:
                    for lr in torch_config.LR:
                        print(f'Grid search: {iteration+1}/{iterations}')
                        print(f'Training CNN with params:\ncnn_depth: {cnn_depth}\nfirst_channel: {first_channel}\nbatch_norm:{batch_norm}\ndrop_out: {drop_out}\nlr: {lr}')
                        unet = UNET(3, first_channel, 1, batch_norm=batch_norm, drop_out=drop_out,
                                    downhill=cnn_depth, padding=0)
                        opt = SGD(unet.parameters(), lr=lr, momentum=0.8, nesterov=False, weight_decay=1e-5)
                        checkpoint = {'model': unet, 'state_dict': None, 'optimizer': None,
                                      'loss': 10000, 'metric': 0}
                        history = training(unet, opt, checkpoint)
                        if best_model['loss'] > min(history['valid_loss']):
                            best_model['metric'] = max(history['valid_metric'])
                            best_model['cnn_depth'] = cnn_depth
                            best_model['first_channel'] = first_channel
                            best_model['batch_norm'] = batch_norm
                            best_model['drop_out'] = drop_out
                            best_model['lr'] = lr
                        print(best_model)
                        iteration += 1

    with open('best_model_sgd.json', 'w', encoding='utf-8') as file:
        json.dump(best_model, file, indent=4)

    return best_model


def training(unet, opt, checkpoint):
    unet = unet.to(torch_config.DEVICE)
    loss_func = DiceBCELoss()  # BCEWithLogitsLoss()

    train_load, test_load, train_num, test_num = get_data()
    train_step = math.ceil(train_num / torch_config.BATCH_SIZE)
    print(f'Train steps: {train_step}')
    test_step = math.ceil(test_num / torch_config.BATCH_SIZE)
    print(f'Test steps: {test_step}')

    history = {'train_metric': [], 'train_loss': [], 'valid_metric': [], 'valid_loss': [], 'valid_accuracy': [],
               'valid_precision': [], 'valid_recall': []}

    # print('Training the network...')
    start_time = time.time()
    epoch_no_improve = 0
    for epoch in tqdm(range(torch_config.NUM_EPOCHS)):
        unet.train()

        total_train_loss = 0
        total_test_loss = 0
        total_train_metric = 0
        total_test_metric = 0

        total_accuracy = 0
        total_precision = 0
        total_recall = 0

        for (i, (x, y)) in enumerate(train_load):
            (x, y) = (x.to(torch_config.DEVICE), y.to(torch_config.DEVICE))

            opt.zero_grad()
            pred = unet(x)

            loss = loss_func.forward(pred, y)
            total_train_loss += loss.item()

            dm = DiceMetric()
            metric = dm.forward(pred, y)
            total_train_metric += metric.item()

            loss.backward()
            opt.step()

            print(
                f'Training batch {i + 1}. Loss: {np.round(loss.item(), 4)}. Metric of Dice: {np.round(metric.item(), 4)}')

        with torch.no_grad():
            unet.eval()

            for (x, y) in test_load:
                (x, y) = (x.to(torch_config.DEVICE), y.to(torch_config.DEVICE))

                pred = unet(x)

                total_test_loss += loss_func.forward(pred, y).item()
                dm = DiceMetric()
                metric = dm.forward(pred, y)
                total_test_metric += metric.item()

                ba = BinaryAccuracy()
                ba.to(device=torch_config.DEVICE)
                metric = ba(pred, y)
                total_accuracy += metric.item()

                bp = BinaryPrecision()
                bp.to(device=torch_config.DEVICE)
                metric = bp(pred, y)
                total_precision += metric.item()

                br = BinaryRecall()
                br.to(device=torch_config.DEVICE)
                metric = br(pred, y).to(torch_config.DEVICE)
                total_recall += metric.item()

        avg_train_loss = total_train_loss / train_step
        avg_test_loss = total_test_loss / test_step
        avg_train_metric = total_train_metric / train_step
        avg_test_metric = total_test_metric / test_step

        avg_test_accuracy = total_accuracy / test_step
        avg_test_precision = total_precision / test_step
        avg_test_recall = total_recall / test_step

        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_test_loss)
        history['train_metric'].append(avg_train_metric)
        history['valid_metric'].append(avg_test_metric)

        history['valid_accuracy'].append(avg_test_accuracy)
        history['valid_precision'].append(avg_test_precision)
        history['valid_recall'].append(avg_test_recall)

        print(f'EPOCH: {epoch + 1}/{torch_config.NUM_EPOCHS}')
        print(f'train loss: {avg_train_loss}\ntrain metric: {avg_train_metric}\ntest loss: {avg_test_loss}\n'
              f'test metric: {avg_test_metric}\ntest accuracy: {avg_test_accuracy}\ntest precision: {avg_test_precision}\ntest recall: {avg_test_recall}')

        if avg_test_loss < checkpoint['loss']:  # test
            epoch_no_improve = 0
            checkpoint['state_dict'] = unet.state_dict()
            checkpoint['optimizer'] = opt.state_dict()
            checkpoint['loss'] = avg_test_loss  # test
            checkpoint['metric'] = avg_test_metric
            checkpoint['precision'] = avg_test_precision
            checkpoint['recall'] = avg_test_recall
            checkpoint['accuracy'] = avg_test_accuracy
        else:
            epoch_no_improve += 1
        if epoch_no_improve >= torch_config.EPOCHS_NO_IMPROVE:  # early stop
            print(f'Early stop: {epoch_no_improve} epochs without improve')
            break

    stop_time = time.time()
    print(f'Training stopped, time: {np.round((stop_time - start_time) / 60, 2)} min')
    print('-' * 150)
    return history, checkpoint


def plot_loss(history: dict, name_file: str):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['valid_loss'], label='test_loss')
    plt.title('Training Loss (BCE + Dice)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(torch_config.PATH_OUTPUT, name_file.split(sep='.pth')[0]+'.png'), dpi=500)
    plt.close()


if __name__ == '__main__':
    hyper_best_dict, hyper_total_dict, history_train = grid_search_adam()
    # hyper_best_dict, hyper_total_dict = grid_search_rmsprop()
    # hyper_dict = grid_search_sgd()
    # history_1, best_dict = training()
    # plot_loss(history=history_train)
