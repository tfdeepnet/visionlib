from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import time

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensor


train_csv = "cifar10/train.csv"
test_csv = "cifar10/test.csv"

train_folder = "cifar10/train"
test_folder = "cifar10/test"


class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self,datafolder , file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.datafolder = datafolder

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        start = time.time()
        label = self.labels[idx]
        file_path = os.path.join(self.datafolder , self.file_paths[idx])

        # Read an image with OpenCV
        image = cv2.imread(file_path)
        #print("fp {} img {}".format(file_path , image))
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        end  = time.time()
        time_spent = (end-start)/60
        print(f"{time_spent:.3} minutes")
        return image, label




def getfdatadf(datafolder , file_path):
    file_path = os.path.join(datafolder ,file_path)
    return pd.read_csv(file_path)

def loadalbumentationdata(datafolder , batch_size ):

    train_df = getfdatadf(datafolder , train_csv)
    test_df = getfdatadf(datafolder , test_csv)

    X_train =train_df["image_path"]
    Y_train =train_df["label"]

    X_test =test_df["image_path"]
    Y_test =test_df["label"]

    albumentations_transform_train = A.Compose([
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.9, rotate_limit=10, p=0.3),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensor()
    ])

    albumentations_transform_test = A.Compose([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensor()
    ])

    albumentations_train_dataset = AlbumentationsDataset(
        datafolder,
        file_paths=X_train,
        labels=Y_train,
        transform=albumentations_transform_train,
    )

    albumentations_test_dataset = AlbumentationsDataset(
        datafolder,
        file_paths=X_test,
        labels=Y_test,
        transform=albumentations_transform_test,
    )

    train_loader = torch.utils.data.DataLoader(albumentations_train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(albumentations_test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=2)

    return train_loader, test_loader
