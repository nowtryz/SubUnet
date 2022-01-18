import torch
import torch.nn as nn


def pooling_module(size: int, in_channels, input_size=32, module_count=4):
    out_channels = in_channels // module_count
    ratio = input_size // size
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=ratio),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        # nn.Dropout(p=, inplace=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=ratio, mode='bilinear', align_corners=True)
    )


class PPM(nn.Module):
    def __init__(self, in_channels):
        super(PPM, self).__init__()
        self.pool_mod1 = pooling_module(size=16, in_channels=in_channels)
        self.pool_mod2 = pooling_module(size=8, in_channels=in_channels)
        self.pool_mod3 = pooling_module(size=2, in_channels=in_channels)
        self.pool_mod4 = pooling_module(size=1, in_channels=in_channels)

    def forward(self, x):
        return torch.cat([x,
                          self.pool_mod1(x),
                          self.pool_mod2(x),
                          self.pool_mod3(x),
                          self.pool_mod4(x)], dim=1)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        # nn.Dropout(p=, inplace=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        # nn.Dropout(p=, inplace=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, avg_pooling=True):
        super(Down, self).__init__()
        self.conv = double_conv(in_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2) if avg_pooling else nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, xi, x):
        x = self.deconv(x)
        x = torch.cat([xi, x], dim=1)
        return self.conv(x)


def output1(n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.Conv2d(64, n_classes, kernel_size=1)
    )


def output2(n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.Conv2d(64, n_classes, kernel_size=1)
    )


def output3(n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.Conv2d(64, n_classes, kernel_size=1)
    )


class SubUNet(nn.Module):
    def __init__(self, n_channels, n_classes, deeply_supervised=True, avg_pooling=True):
        """

        :param n_channels: number of input channels
        :param n_classes:  number of classes to predict
        :param deeply_supervised: weather or not to enable deep supervision
        :param avg_pooling: if false, the model will switch to MaxPooling for the UNet encoder
        """
        super(SubUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deeply_supervised = deeply_supervised

        self.down1 = Down(self.n_channels, 64, avg_pooling=avg_pooling)
        self.down2 = Down(64, 128, avg_pooling=avg_pooling)
        self.down3 = Down(128, 256, avg_pooling=avg_pooling)
        self.conv1 = double_conv(256, 512)
        self.ppm = PPM(512)
        self.conv2 = double_conv(1024, 512)
        self.output1 = output1(n_classes)
        self.up1 = Up(512, 256)
        self.output2 = output2(n_classes)
        self.up2 = Up(256, 128)
        self.output3 = output3(n_classes)
        self.up3 = Up(128, 64)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)
        x = self.conv1(x)
        x = self.ppm(x)
        x = self.conv2(x)

        # First deeply-supervised output
        if self.training and self.deeply_supervised:
            out1 = self.output1(x)

        x = self.up1(x3, x)

        # Second deeply-supervised output
        if self.training and self.deeply_supervised:
            out2 = self.output2(x)

        x = self.up2(x2, x)

        # Third deeply-supervised output
        if self.training and self.deeply_supervised:
            out3 = self.output3(x)

        x = self.up3(x1, x)
        x = self.out_conv(x)

        if self.training and self.deeply_supervised:
            return x, out1, out2, out3

        return x
