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
from Datasets.lip import LIPWithClass
from matplotlib import pyplot as plt
from Net.pspnet import PSPNet


# Models
models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend, gpu=False):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    if gpu:
        net = net.cuda()
    return net, epoch


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
    parser.add_argument('-n', '--num-class', help='Set number of segmentation classes', default=20, type=int)
    parser.add_argument('-be', '--backend', help='Set Feature extractor', default='densenet', type=str)
    parser.add_argument('-s', '--snapshot', help='Set path to pre-trained weights', default=None, type=str)
    parser.add_argument('-g', '--gpu', help='Set gpu [True / False]', default=False, type=bool)

    # Mutually Exclusive Group 1 (Train / Eval)
    train_eval_parser = parser.add_mutually_exclusive_group(required=False)
    train_eval_parser.add_argument('-t', '--train', dest='train', help='Set to train mode', action='store_true')
    train_eval_parser.add_argument('-ev', '--eval', dest='train', help='Set to eval mode', action='store_false')
    parser.set_defaults(train=True)

    args = parser.parse_args()
    return args


def get_dataloader(data_path, train=True, batch_size=32, shuffle=False, num_class=20):
    return DataLoader(
        dataset=LIPWithClass(
            par_path=data_path,
            transform=get_transform(),
            train=train,
            num_class=num_class
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
        img, gt, gt_cls = data
        pred = torch.argmax(model_ft(img)['out'], dim=1)

        print(img.shape, gt.shape, pred.shape) # Debug
        print(np.unique(gt), np.unique(pred)) # Debug
        plot_img_gt_pred(img[0], gt[0], pred[0]) # Debug


if __name__ == '__main__':
    # Parse Arguments
    args = parse_arguments()

    # Make directory to store trained weights
    models_path = os.path.join('./checkpoints', args.backend)
    os.makedirs(models_path, exist_ok=True)

    train_loader = get_dataloader(args.data_path, train=args.train, batch_size=args.batch_size, num_class=args.num_class)

    # Debug
    for data in train_loader:
        x, y, cls = data
        break

    # run_trained_model(models['torch_resnet50'], train_loader)


