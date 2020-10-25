import torch
import torchvision.utils as vutils
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import argparse

dataset_dir = '../../datasets/voc/VOC2012'
segmentation_class_dir = dataset_dir + '/SegmentationClass'  # ground truth:png
images_dir = dataset_dir + '/JPEGImages'  # image:jpg
val_txt = dataset_dir + '/ImageSets/Segmentation/val.txt'

with open(val_txt, 'r') as fin:
    vals = fin.readlines()

vals = [val.strip() for val in vals]


def parse_args():
    parser = argparse.ArgumentParser(description='Visually display predicted results')
    parser.add_argument('--model', type=str, default='fcn32s',
                        choices=['fcn32s', 'fcn16s', 'fcn8s',
                                 'fcn', 'psp', 'deeplabv3', 'deeplabv3_plus',
                                 'danet', 'denseaspp', 'bisenet',
                                 'encnet', 'dunet', 'icnet',
                                 'enet', 'ocnet', 'ccnet', 'psanet',
                                 'cgnet', 'espnet', 'lednet', 'dfanet'],
                        help='model name (default: fcn32s)')
    return parser.parse_args()


def write_tensorboard(args):
    preds_dir = '../../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)  # pred:png
    runs_folder = '../runs/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
    writer = SummaryWriter(runs_folder)
    i = 0
    subplot_index = 0
    for val in vals:
        i = i + 1
        if i % 250 == 0:
            image_np = plt.imread(images_dir + '/{}.jpg'.format(val))
            pred_np = plt.imread(preds_dir + '/{}.png'.format(val))
            truth_np = plt.imread(segmentation_class_dir + '/{}.png'.format(val))
            figure = plt.figure(num="predicted result")
            plt.subplots_adjust(left=0.0, bottom=0.0, top=0.95, right=0.95)
            subplot_index += 1
            plt.subplot(5, 3, subplot_index)
            plt.imshow(image_np)
            plt.axis('off')
            if subplot_index == 1:
                plt.title('image')

            subplot_index += 1
            plt.subplot(5, 3, subplot_index)
            plt.imshow(pred_np)
            plt.axis('off')
            if subplot_index == 2:
                plt.title('200 epochs')

            subplot_index += 1
            plt.subplot(5, 3, subplot_index)
            plt.imshow(truth_np)
            plt.axis('off')
            if subplot_index == 3:
                plt.title('ground truth')

            # row = np.concatenate((image_np, pred_np, truth_np), axis=0)
            # writer.add_image('row', row, dataformats='HWC')
            # row = np.stack((image_np, pred_np, truth_np), axis=0)
            # writer.add_images('result', row, dataformats='NHWC')
            # writer.add_image('image', image_np, dataformats='HWC')
            # writer.add_image('image', pred_np, dataformats='HWC')
            # writer.add_image('ground truth', truth_np, dataformats='HWC')

    writer.add_figure('{}_{}_{}'.format(args.model, args.backbone, args.dataset), figure)
    writer.close()


def write_fcn32s(args):
    preds_dir = '../../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)  # pred:png
    preds_50epochs_dir = '../../runs/pred_pic/{}_{}_{}_50epochs'.format(args.model, args.backbone, args.dataset)  # pred:png
    runs_folder = '../runs/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
    writer = SummaryWriter(runs_folder)
    i = 0
    subplot_index = 0
    for val in vals:
        i += 1
        if i % 250 == 0:
            image_np = plt.imread(images_dir + '/{}.jpg'.format(val))
            pred_np = plt.imread(preds_dir + '/{}.png'.format(val))
            pred_50epochs_np = plt.imread(preds_50epochs_dir + '/{}.png'.format(val))
            truth_np = plt.imread(segmentation_class_dir + '/{}.png'.format(val))
            figure = plt.figure(num="predicted result")
            plt.subplots_adjust(left=0.0, bottom=0.0, top=0.95, right=0.95)
            subplot_index += 1
            plt.subplot(5, 4, subplot_index)
            plt.imshow(image_np)
            plt.axis('off')
            if subplot_index == 1:
                plt.title('image')

            subplot_index += 1
            plt.subplot(5, 4, subplot_index)
            plt.imshow(pred_50epochs_np)
            plt.axis('off')
            if subplot_index == 2:
                plt.title('50 epochs')

            subplot_index += 1
            plt.subplot(5, 4, subplot_index)
            plt.imshow(pred_np)
            plt.axis('off')
            if subplot_index == 3:
                plt.title('200 epochs')

            subplot_index += 1
            plt.subplot(5, 4, subplot_index)
            plt.imshow(truth_np)
            plt.axis('off')
            if subplot_index == 4:
                plt.title('ground truth')

    writer.add_figure('{}_{}_{}'.format(args.model, args.backbone, args.dataset), figure)
    writer.close()


def write_together(args):
    preds_fcn8s_dir = '../../runs/pred_pic/fcn8s_{}_{}'.format(args.backbone, args.dataset)  # pred:png
    preds_fcn16s_dir = '../../runs/pred_pic/fcn16s_{}_{}'.format(args.backbone, args.dataset)  # pred:png
    preds_fcn32s_dir = '../../runs/pred_pic/fcn32s_{}_{}'.format(args.backbone, args.dataset)  # pred:png
    preds_50epochs_dir = '../../runs/pred_pic/fcn32s_{}_{}_50epochs'.format(args.backbone, args.dataset)  # pred:png
    runs_folder = '../runs/together_{}_{}'.format(args.backbone, args.dataset)
    writer = SummaryWriter(runs_folder)
    i = 0
    subplot_index = 0
    for val in vals:
        i += 1
        if 724 < i < 730:
            image_np = plt.imread(images_dir + '/{}.jpg'.format(val))
            pred_fcn8s_np = plt.imread(preds_fcn8s_dir + '/{}.png'.format(val))
            pred_fcn16s_np = plt.imread(preds_fcn16s_dir + '/{}.png'.format(val))
            pred_fcn32s_np = plt.imread(preds_fcn32s_dir + '/{}.png'.format(val))
            pred_50epochs_np = plt.imread(preds_50epochs_dir + '/{}.png'.format(val))
            truth_np = plt.imread(segmentation_class_dir + '/{}.png'.format(val))

            figure = plt.figure(num="predicted result")
            plt.subplots_adjust(left=0.0, bottom=0.0, top=0.95, right=0.95)
            subplot_index += 1
            plt.subplot(5, 6, subplot_index)
            plt.imshow(image_np)
            plt.axis('off')
            if subplot_index == 1:
                plt.title('image')

            subplot_index += 1
            plt.subplot(5, 6, subplot_index)
            plt.imshow(pred_50epochs_np)
            plt.axis('off')
            if subplot_index == 2:
                plt.title('fcn32s(50)')

            subplot_index += 1
            plt.subplot(5, 6, subplot_index)
            plt.imshow(pred_fcn32s_np)
            plt.axis('off')
            if subplot_index == 3:
                plt.title('fcn32s(200)')

            subplot_index += 1
            plt.subplot(5, 6, subplot_index)
            plt.imshow(pred_fcn16s_np)
            plt.axis('off')
            if subplot_index == 4:
                plt.title('fcn16s(200)')

            subplot_index += 1
            plt.subplot(5, 6, subplot_index)
            plt.imshow(pred_fcn8s_np)
            plt.axis('off')
            if subplot_index == 5:
                plt.title('fcn8s(200)')

            subplot_index += 1
            plt.subplot(5, 6, subplot_index)
            plt.imshow(truth_np)
            plt.axis('off')
            if subplot_index == 6:
                plt.title('ground truth')

    writer.add_figure('together_{}_{}'.format(args.backbone, args.dataset), figure)
    writer.close()

def write_deeplabv3(args):
    preds_dir = '../../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)  # pred:png
    runs_folder = '../runs/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
    writer = SummaryWriter(runs_folder)
    i = 0
    subplot_index = 0
    for val in vals:
        i += 1
        if i % 250 == 0:
            image_np = plt.imread(images_dir + '/{}.jpg'.format(val))
            pred_np = plt.imread(preds_dir + '/{}.png'.format(val))
            pred_50epochs_np = plt.imread(preds_50epochs_dir + '/{}.png'.format(val))
            truth_np = plt.imread(segmentation_class_dir + '/{}.png'.format(val))
            figure = plt.figure(num="predicted result")
            plt.subplots_adjust(left=0.0, bottom=0.0, top=0.95, right=0.95)
            subplot_index += 1
            plt.subplot(5, 4, subplot_index)
            plt.imshow(image_np)
            plt.axis('off')
            if subplot_index == 1:
                plt.title('image')

            subplot_index += 1
            plt.subplot(5, 4, subplot_index)
            plt.imshow(pred_50epochs_np)
            plt.axis('off')
            if subplot_index == 2:
                plt.title('50 epochs')

            subplot_index += 1
            plt.subplot(5, 4, subplot_index)
            plt.imshow(pred_np)
            plt.axis('off')
            if subplot_index == 3:
                plt.title('200 epochs')

            subplot_index += 1
            plt.subplot(5, 4, subplot_index)
            plt.imshow(truth_np)
            plt.axis('off')
            if subplot_index == 4:
                plt.title('ground truth')

    writer.add_figure('{}_{}_{}'.format(args.model, args.backbone, args.dataset), figure)
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    args.backbone = 'resnet101'
    args.dataset = 'pascal_voc'
    write_together(args)

    '''if args.model == 'fcn32s':
        write_fcn32s(args)
    else:
        write_tensorboard(args)'''








