# Model inspired by https://arxiv.org/pdf/1511.00561v3.pdf
#                   https://www.semanticscholar.org/paper/SegNetRes-CRF%3A-A-Deep-Convolutional-Encoder-Decoder-Junior-Medeiros/991673d4f9dd08893723549ff3ea866b2dc18047
import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    """(convolution => [BN] => ReLU) * 2"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def triple_conv(in_channels, out_channels):
    """(convolution => [BN] => ReLU) * 3"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def max_pool():
    return nn.MaxPool2d(kernel_size=2, return_indices=True)

def down(in_channels, out_channels, three_conv=True):
    modules = [triple_conv(in_channels, out_channels) if three_conv else double_conv(in_channels, out_channels)]
    modules.append(max_pool())
    return nn.Sequential(*modules)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, three_conv=True):
        super().__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2)
        self.conv = triple_conv(in_channels, out_channels) if three_conv else double_conv(in_channels, out_channels)

    def forward(self, x, indices):
        x = self.max_unpool(x, indices)
        return self.conv(x)


class SegNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SegNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = down(n_channels, 64, False)
        self.down2 = down(64, 128, False)
        self.down3 = down(128, 256, True)
        self.down4 = down(256, 512, True)
        self.down5 = down(512, 1024, True)

        self.up1 = Up(1024, 512, True)
        self.up2 = Up(512, 256, True)
        self.up3 = Up(256, 128, True)
        self.up4 = Up(128, 64, False)
        self.up5 = Up(64, n_classes, False)

    def forward(self, x):
        x, i1 = self.down1(x)
        x, i2 = self.down2(x)
        x, i3 = self.down3(x)
        x, i4 = self.down4(x)
        x, i5 = self.down5(x)

        x = self.up1(x, i5)
        x = self.up2(x, i4)
        x = self.up3(x, i3)
        x = self.up4(x, i2)
        x = self.up5(x, i1)
        return x
