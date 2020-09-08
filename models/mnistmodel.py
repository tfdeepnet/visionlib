from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class InitModelParams:

    def __init__(self, num_splits = 1):
        self.modelHyperParamDict = {"GBN_NUM_SPLITS": num_splits}

    def getModelHyperParamDict(self):
        return self.modelHyperParamDict

    def clearModelHyperParamDict(self):
        self.modelHyperParamDict.clear()


imp = InitModelParams()




class GhostBatchNorm(nn.BatchNorm2d):
    """
    From : https://github.com/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb

    Batch norm seems to work best with batch size of around 32. The reasons presumably have to do
    with noise in the batch statistics and specifically a balance between a beneficial regularising effect
    at intermediate batch sizes and an excess of noise at small batches.

    Our batches are of size 512 and we can't afford to reduce them without taking a serious hit on training times,
    but we can apply batch norm separately to subsets of a training batch. This technique, known as 'ghost' batch
    norm, is usually used in a distributed setting but is just as useful when using large batches on a single node.
    It isn't supported directly in PyTorch but we can roll our own easily enough.
    """

    def __init__(self, num_features, num_splits, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super(GhostBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat \
                (self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat \
                (self.num_splits)
        return super(GhostBatchNorm, self).train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


class BNNet(nn.Module):
    def __init__(self):
        super(BNNet, self).__init__()
        # Input Block
        c_in = 1
        c_out = 8
        # print("running BN network")
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )  # output_size = 26

        # CONVOLUTION BLOCK 1
        c_in = 8
        c_out = 8
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )  # output_size = 24

        c_in = 8
        c_out = 8
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )  # output_size = 22

        # TRANSITION BLOCK 1

        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 11

        c_in = 8
        c_out = 16
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )  # output_size = 22

        # CONVOLUTION BLOCK 2
        c_in = 16
        c_out = 16

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )  # output_size = 9
        c_in = 16
        c_out = 16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )  # output_size = 7

        c_in = 16
        c_out = 10
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), padding=0, bias=False),
            # nn.ReLU() NEVER!
        )  # output_size = 1

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)

        x = self.convblock9(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class GBNNet(nn.Module):
    def __init__(self, gbn_splits=0):
        super(GBNNet, self).__init__()
        self.GBN_NUM_SPLITS = gbn_splits

        # Input Block
        c_in = 1
        c_out = 8
        # print("running GBN network")
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            GhostBatchNorm(c_out, num_splits=self.GBN_NUM_SPLITS, weight=False),
            nn.ReLU()
        )  # output_size = 26

        # CONVOLUTION BLOCK 1
        c_in = 8
        c_out = 8
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            GhostBatchNorm(c_out, num_splits=self.GBN_NUM_SPLITS, weight=False),
            nn.ReLU()
        )  # output_size = 24

        c_in = 8
        c_out = 8
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            GhostBatchNorm(c_out, num_splits=self.GBN_NUM_SPLITS, weight=False),
            nn.ReLU()
        )  # output_size = 22

        # TRANSITION BLOCK 1

        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 11

        c_in = 8
        c_out = 16
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            GhostBatchNorm(c_out, num_splits=self.GBN_NUM_SPLITS, weight=False),
            nn.ReLU()
        )  # output_size = 22

        # CONVOLUTION BLOCK 2
        c_in = 16
        c_out = 16

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            GhostBatchNorm(c_out, num_splits=self.GBN_NUM_SPLITS, weight=False),
            nn.ReLU()
        )  # output_size = 9
        c_in = 16
        c_out = 16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), padding=0, bias=False),
            GhostBatchNorm(c_out, num_splits=self.GBN_NUM_SPLITS, weight=False),
            nn.ReLU()
        )  # output_size = 7

        c_in = 16
        c_out = 10
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), padding=0, bias=False),
            # nn.ReLU() NEVER!
        )  # output_size = 1

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)

        x = self.convblock9(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
