import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets
import albumentations as A
import albumentations.pytorch as tfm

def loaddata(batch_size , datasetname = "Cifar10"):

    #train_transforms = transforms.Compose([
        #transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1)),
        #transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_transforms = A.Compose([
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.9, rotate_limit=10, p=0.3),
        #tfm.ToTensorV2(),
        A.pytorch.ToTensor(),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1)),
        #transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transforms = A.Compose([
        #tfm.ToTensorV2(),
        A.pytorch.ToTensorV2(),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #transforms.Compose(
        #[transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if (datasetname == "Cifar10"):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader