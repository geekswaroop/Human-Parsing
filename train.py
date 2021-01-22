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
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('-t', '--train', dest='train', action='store_true')
    feature_parser.add_argument('-ev', '--eval', dest='train', action='store_false')
    parser.set_defaults(train=True)

    args = parser.parse_args()
    return args


def get_dataloader(data_path, train=True):
    return DataLoader(
        dataset=LIP(
            par_path=data_path,
            transform=get_transform(),
            train=train
        ), 
        batch_size=1, 
        shuffle=False
    )
   

if __name__ == '__main__':
    args = parse_arguments()
    train_loader = get_dataloader(args.data_path, train=args.train)
    model_ft = models.segmentation.fcn_resnet50(pretrained=True, progress=True, num_classes=21)
    for data in train_loader:
        X, Y = data
        print(torch.argmax(model_ft(X)['out'], dim=1).shape)

        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(3,1) 

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        axarr[0].imshow(X[0].permute(1, 2, 0))
        axarr[1].imshow(Y.permute(1, 2, 0))
        axarr[2].imshow(torch.argmax(model_ft(X)['out'], dim=1).permute(1, 2, 0))
        plt.show()


    print('Dataset Loaded') # Log


