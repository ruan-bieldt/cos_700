import torch.nn as nn
import torch
import torch.nn.functional as F


class LargeResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start=False, stride=1):
        super(LargeResidualBlock, self).__init__()

        if start:
            self.conv1 = nn.Conv2d(in_channels*2, in_channels,
                                   kernel_size=1, stride=stride, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels*2, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(out_channels, in_channels,
                                   kernel_size=1, stride=stride, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.projection = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # If the stride is not 1 or the number of input channels is different from the output channels,
        # we need to adjust the dimensions using a 1x1 convolutional layer (projection)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.projection(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the stride is not 1 or the number of input channels is different from the output channels,
        # we need to adjust the dimensions using a 1x1 convolutional layer (projection)
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.projection(x)

        out += identity
        out = self.relu(out)

        return out


def conv_block(in_channels, out_channels, downsample=False):
    stride_l = 1
    if downsample:
        stride_l = 2
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride_l),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
