import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.kl_resnet import *
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def getclassvalidationAccuracy(model, testdata, classes, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testdata:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def view_misclassified_images(misclassified_images , classes):
    fig = plt.figure(figsize=(8, 8))
    for i in range(25):
        sub = fig.add_subplot(5, 5, i + 1)
        # imshow(misclassified_images[i][0].cpu())
        img = misclassified_images[i][0].cpu()
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='none')

        sub.set_title("P={}, A={}".format(str(classes[misclassified_images[i][1].data.cpu().numpy()]),
                                          str(classes[misclassified_images[i][2].data.cpu().numpy()])))

    plt.tight_layout()


def get_validation_result_and_misclassifiedimages(model,
                                                  device,
                                                  classes,
                                                  test_loader,
                                                  total_images = 25 ,
                                                  printClassAccuracy=True,
                                                  printValidationAccuracy=True):
    misclassified_images = []
    correct = 0
    total = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                if (len(misclassified_images) < total_images and predicted[i] != labels[i]):
                    misclassified_images.append([images[i], predicted[i], labels[i]])
            if (len(misclassified_images) > total_images):
                    break

            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    if (printClassAccuracy):
        print("Class accuracy \n")
        for i in range(10):
            print('Accuracy of %5s : %2d %% \n' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

    if (printValidationAccuracy):
        print('\n Accuracy of the network on the 10000 test images: %.2f %%' % (
                100 * correct / total))

    return misclassified_images


def plot_train_vs_test_accuracy(epochs, train_acc, test_acc):
    train_range = range(1, epochs + 1)
    plt.plot(train_range, train_acc, 'g', label='Training accuracy')
    plt.plot(train_range, test_acc, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
