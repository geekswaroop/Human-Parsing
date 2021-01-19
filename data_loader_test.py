import os
import imageio as img
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Setting paths
PAR_PATH = os.path.join(os.path.expanduser('~'), 'Desktop/anirudh/ML/Projects/HumanParsingIEEE/Datasets/LIP')
TRAIN_IMAGES_PATH = os.path.join(PAR_PATH, 'TrainVal_images', 'train_images')
TRAIN_IMAGES_ANNOTATIONS_PATH = os.path.join(PAR_PATH, 'TrainVal_parsing_annotations', 'train_segmentations')


# Funtion to return sorted list of all files in a directory
def list_dir(path):
    files_list = []

    for files in os.listdir(path):
        files_list.append(files)

    files_list.sort() # Sorting according to filename
    return files_list


# List train images and annotations filenames (Checks for mismatches)
def list_train_dataset(train_images_path, train_images_annotations_path):

    X = list_dir(train_images_path) # Train images
    Y = list_dir(train_images_annotations_path) # Train image annotations
    
    # Checking if X, Y files match
    for x, y in zip(X, Y):
        if x[0:-3] != y[0: -3]: 
            raise Exception('Error : Mismatched')

    return X, Y

    
# Loads train dataset
def load_train_dataset():
    transform = get_transform()

    # Loading train_dataset_names
    train_images_files, train_images_annotations_files = list_train_dataset(TRAIN_IMAGES_PATH, TRAIN_IMAGES_ANNOTATIONS_PATH)
    image_count = 1

    train_images = []
    train_images_annotations = []

    for train_image, train_image_annotation in zip(train_images_files, train_images_annotations_files):
        img = Image.open(os.path.join(TRAIN_IMAGES_PATH, train_image))
        gt = Image.open(os.path.join(TRAIN_IMAGES_ANNOTATIONS_PATH, train_image_annotation))

        train_images.append(transform['img'](img))
        train_images_annotations.append(transform['gt'](gt))

        # train_images.append(Image.open(os.path.join(TRAIN_IMAGES_PATH, train_image)))
        # train_images_annotations.append(Image.open(os.path.join(TRAIN_IMAGES_ANNOTATIONS_PATH, train_image_annotation)))
        
        image_count += 1
        if image_count > 100: 
            break

    return train_images, train_images_annotations

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

class LIP(Dataset):

    def __init__(self, transform):
        train_images, train_images_annotations = load_train_dataset()
        print(len(train_images))

        self.len = len(train_images)
        self.x_data = train_images
        self.y_data = train_images_annotations
        print('completed')

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len



if __name__ == '__main__':
    transform = get_transform()
    dataset = LIP(transform=transform)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(train_loader, 0):
        x, y = data

        print(x)
        print(np.unique(y[0]))
