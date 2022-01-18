# Model from https://www.researchgate.net/figure/Architecture-of-the-FCN-VGG19-adapted-from-Long-et-al-2015-which-learns-to-combine_fig1_331258180

import torch
import torch.nn as nn


def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def double_conv(in_channels, out_channels, maxpool=True):
    modules = [
        single_conv(in_channels, out_channels),
        single_conv(out_channels, out_channels)
    ]
    if maxpool:
        modules.append(nn.MaxPool2d(2))
    return nn.Sequential(*modules)

def quad_conv(in_channels, out_channels):
    return nn.Sequential(
        single_conv(in_channels, out_channels),
        single_conv(out_channels, out_channels),
        single_conv(out_channels, out_channels),
        single_conv(out_channels, out_channels),
        nn.MaxPool2d(2)
    )


class Up(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = single_conv(2*n_classes, n_classes)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class VGG19(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(VGG19, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv123 = nn.Sequential(
            double_conv(self.n_channels, 64),
            double_conv(64, 128),
            quad_conv(128, 256)
        )
        self.conv4 = quad_conv(256, 512)
        self.conv56 = nn.Sequential(
            quad_conv(512, 512),
            double_conv(512, 4096, maxpool=False),
            single_conv(4096, self.n_classes)
        )

        self.extract1 = single_conv(256, self.n_classes)
        self.extract2 = single_conv(512, self.n_classes)

        self.up = Up(self.n_classes)
        self.output = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv123(x)
        x1 = self.extract1(x)
        x = self.conv4(x)
        x2 = self.extract2(x)
        x = self.conv56(x)

        x = self.up(x, x2)
        x = self.up(x, x1)
        x = self.output(x)
        return x
