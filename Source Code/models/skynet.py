import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.unet import double_conv as unet_double_conv, Up as UNetUp, down as unet_down


class UNet(nn.Module):
    """docstring for UNet"""

    def __init__(self, n_channels, n_classes, bilinear=True, crop=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.crop = crop

        self.in_conv = unet_double_conv(n_channels, 128)
        self.down1 = unet_down(128, 256)
        self.down2 = unet_down(256, 512)
        self.down3 = unet_down(512, 1024)
        factor = 2 if self.bilinear else 1
        self.down4 = unet_down(1024, 2048 // factor)
        self.up1 = UNetUp(2048, 1024 // factor, bilinear=self.bilinear, crop=self.crop)
        self.up2 = UNetUp(1024, 512 // factor, bilinear=self.bilinear, crop=self.crop)
        self.up3 = UNetUp(512, 256 // factor, bilinear=self.bilinear, crop=self.crop)
        self.up4 = UNetUp(256, 128, bilinear=self.bilinear, crop=self.crop)
        self.out_conv = nn.Conv2d(128, n_classes, 1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_out1 = self.up1(x5, x4)
        x_out2 = self.up2(x_out1, x3)
        x_out3 = self.up3(x_out2, x2)
        x = self.up4(x_out3, x1)
        logits = self.out_conv(x)
        return x_out1, x_out2, x_out3, logits


def PSP_single_conv(in_channels, out_channels, pooling=True):
    """(convolution => [BN] => ReLU)"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2) if pooling else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    )


class PSP_PPM(nn.Module):
    def __init__(self, in_channels, out_channels, nb_layer=3):
        super(PSP_PPM, self).__init__()
        self.pools = []
        self.unets = []
        self.unpools = []
        self.nb_layer = nb_layer
        for layer in range(self.nb_layer):  # niveaux de poolings du pyramid pooling module
            k = 2 ** (nb_layer - layer)
            self.pools.append(nn.MaxPool2d(kernel_size=k, return_indices=True))
            self.unets.append(UNet(in_channels, out_channels))
            self.unpools.append(nn.MaxUnpool2d(kernel_size=k))
        self.pools = nn.ModuleList(self.pools)
        self.unets = nn.ModuleList(self.unets)
        self.unpools = nn.ModuleList(self.unpools)

    def forward(self, x):
        out = {'final': [], '1': [], '2': [], '3': []}
        for layer in range(self.nb_layer):
            scale_factor = 2 ** (self.nb_layer - layer - 1)
            x_tmp, ind = self.pools[layer](x)
            x_tmp1, x_tmp2, x_tmp3, x_tmp = self.unets[layer](x_tmp)
            x_tmp = self.unpools[layer](x_tmp, ind)
            out['final'].append(x_tmp)
            out['1'].append(F.upsample(x_tmp1, scale_factor=scale_factor))
            out['2'].append(F.upsample(x_tmp2, scale_factor=scale_factor))
            out['3'].append(F.upsample(x_tmp3, scale_factor=scale_factor))
        return torch.cat(out['final'], 1), torch.cat(out['1'], 1), torch.cat(out['2'], 1), torch.cat(out['3'], 1)


class SkyNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SkyNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = PSP_single_conv(self.n_channels, 64, True)
        self.ppm = PSP_PPM(64, 64, nb_layer=3)
        self.out_conv = nn.Sequential(
            PSP_single_conv(256, 64, False),
            nn.Conv2d(64, self.n_classes, kernel_size=3, padding=1),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1536, 64, kernel_size=4, stride=4, padding=0),  # x4
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0),  # x4
            nn.ConvTranspose2d(32, n_classes, kernel_size=2, stride=2, padding=0),  # x2
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 64, kernel_size=4, stride=4, padding=0),  # x4
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0),  # x4
            nn.Conv2d(32, n_classes, kernel_size=3, padding=1),  # x1
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 64, kernel_size=4, stride=4, padding=0),  # x4
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),  # x2
            nn.Conv2d(32, n_classes, kernel_size=3, padding=1),  # x1
        )

    def forward(self, x):
        x = self.in_conv(x)
        xp, x1, x2, x3 = self.ppm(x)
        x = torch.cat([x, xp], dim=1)
        x1 = self.deconv1(x1)
        x2 = self.deconv2(x2)
        x3 = self.deconv3(x3)
        x = self.out_conv(x)
        return x, x1, x2, x3
