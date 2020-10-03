import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

test_losses = []
test_acc = []

def test(testloader, net, criterion , device ):
    correct = 0
    total = 0
    test_loss = 0

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            test_loss += criterion(outputs, labels).item()  # sum up batch loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    testsetsize = len(testloader.dataset)
    test_loss /= testsetsize
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, testsetsize,
        100. * correct / testsetsize))

    test_acc.append(100. * correct / testsetsize)
    return test_losses,test_acc

