import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Setting paths
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

class LIP(Dataset):

    def __init__(self, par_path, transform, is_local_testing=False):

        self.par_path = par_path
        self.transform = transform
        self.is_local_testing = is_local_testing

        self.train_images_path = os.path.join(par_path, 'TrainVal_images', 'train_images')
        self.train_images_annotations_path = os.path.join(par_path, 'TrainVal_parsing_annotations', 'train_segmentations')

        train_images, train_images_annotations = self.load_train_dataset()

        self.len = len(train_images)
        self.x_data = train_images
        self.y_data = train_images_annotations

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

    # Loads train dataset
    def load_train_dataset(self):

        # Loading train_dataset_names
        train_images_files, train_images_annotations_files = self.list_train_dataset()
        image_count = 1

        train_images = []
        train_images_annotations = []

        for train_image, train_image_annotation in zip(train_images_files, train_images_annotations_files):
            img = Image.open(os.path.join(self.train_images_path, train_image))
            gt = Image.open(os.path.join(self.train_images_annotations_path, train_image_annotation))

            train_images.append(transform['img'](img))
            train_images_annotations.append(transform['gt'](gt))
            
            # If local testing load a subset of dataset
            if self.is_local_testing:
                image_count += 1
                if image_count > 100: 
                    break

        return train_images, train_images_annotations


    # List train images and annotations filenames (Checks for mismatches)
    def list_train_dataset(self):

        X = self.list_dir(self.train_images_path) # Train images
        Y = self.list_dir(self.train_images_annotations_path) # Train image annotations
        
        # Checking if X, Y files match
        for x, y in zip(X, Y):
            if x[0:-3] != y[0: -3]: 
                raise Exception('Error : Mismatched')

        return X, Y

    # Funtion to return sorted list of all files in a directory
    def list_dir(self, path):
        files_list = []

        for files in os.listdir(path):
            files_list.append(files)

        files_list.sort() # Sorting according to filename
        return files_list


if __name__ == '__main__':
    transform = get_transform()
    dataset = LIP(par_path=PAR_PATH, transform=transform, is_local_testing=False)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    print('Dataset Loaded') # Log

