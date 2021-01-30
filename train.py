# Importing libraries
import os
import argparse
import numpy as np
import torch 
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from Datasets.lip import LIP
from matplotlib import pyplot as plt


# Models
models = {
    'torch_resnet50': models.segmentation.fcn_resnet50(pretrained=True, progress=True, num_classes=21).eval()
}


def get_transform():
    transform_image_list = [
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    transform_gt_list = [
        transforms.Resize((256, 256), 0),
        transforms.Lambda(lambda img: np.asarray(img, dtype=np.uint8)),
    ]

    data_transforms = {
        'img': transforms.Compose(transform_image_list),
        'gt': transforms.Compose(transform_gt_list),
    }
    return data_transforms


def parse_arguments():
    parser = argparse.ArgumentParser(description='Human Parsing')

    # Add more arguments based on requirements later
    parser.add_argument('-e', '--epochs', help='Set number of train epochs', default=100, type=int)
    parser.add_argument('-b', '--batch-size', help='Set size of the batch', default=32, type=int)
    parser.add_argument('-d', '--data-path', help='Set path of dataset', default='.', type=str)

    # Mutually Exclusive Group 1 (Train / Eval)
    train_eval_parser = parser.add_mutually_exclusive_group(required=False)
    train_eval_parser.add_argument('-t', '--train', dest='train', help='Set to train mode', action='store_true')
    train_eval_parser.add_argument('-ev', '--eval', dest='train', help='Set to eval mode', action='store_false')
    parser.set_defaults(train=True)

    args = parser.parse_args()
    return args


def get_dataloader(data_path, train=True, batch_size=32, shuffle=False):
    return DataLoader(
        dataset=LIP(
            par_path=data_path,
            transform=get_transform(),
            train=train
        ), 
        batch_size=batch_size, 
        shuffle=shuffle
    )
   

def plot_img_gt_pred(img, gt, pred):
    f, axarr = plt.subplots(3,1) 
    axarr[0].imshow(img.permute(1, 2, 0)) # Image (RGB)
    axarr[1].imshow(gt) # Ground truth (Grey Scale)
    axarr[2].imshow(pred) # Prediction (Grey Scale)
    plt.show()


def run_trained_model(model_ft, train_loader):
    for data in train_loader:
        img, gt = data
        pred = torch.argmax(model_ft(img)['out'], dim=1)

        print(img.shape, gt.shape, pred.shape) # Debug
        print(np.unique(gt), np.unique(pred)) # Debug
        plot_img_gt_pred(img[0], gt[0], pred[0]) # Debug


if __name__ == '__main__':
    args = parse_arguments()
    train_loader = get_dataloader(args.data_path, train=args.train, batch_size=args.batch_size)
    run_trained_model(models['torch_resnet50'], train_loader)


