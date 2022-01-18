# Model inspired by https://github.com/milesial/Pytorch-UNet/
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def double_conv(in_channels, out_channels, mid_channels=None):
    """(convolution => [BN] => ReLU) * 2"""
    if not mid_channels:
        mid_channels = out_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def down(in_channels, out_channels):
    """Downscaling with maxpool then double conv"""
    return nn.Sequential(
        nn.MaxPool2d(2),
        double_conv(in_channels, out_channels)
    )


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, crop=False):
        super().__init__()
        self.crop = crop

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = double_conv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if self.crop:
            x2 = transforms.CenterCrop(x1.size()[2:3])(x2)
        else:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        if self.crop:
            x = nn.Upsample(size=(diffY + x1.size()[2], diffX + x1.size()[3]), mode='bilinear', align_corners=True)(x)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, crop=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.crop = crop

        self.in_conv = double_conv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear=self.bilinear, crop=self.crop)
        self.up2 = Up(512, 256 // factor, bilinear=self.bilinear, crop=self.crop)
        self.up3 = Up(256, 128 // factor, bilinear=self.bilinear, crop=self.crop)
        self.up4 = Up(128, 64, bilinear=self.bilinear, crop=self.crop)
        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits
