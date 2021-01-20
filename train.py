# Importing libraries
import os
import argparse
import numpy as np
import torch 
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data_loader import LIP

PAR_PATH = os.path.join(os.path.expanduser('~'), 'Desktop/anirudh/ML/Projects/HumanParsingIEEE/Datasets/LIP')

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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    transform = get_transform()
    dataset = LIP(par_path=args.data_path, transform=transform) 
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for data in train_loader:
        X, Y = data
        print(X, Y)

    print('Dataset Loaded') # Log

