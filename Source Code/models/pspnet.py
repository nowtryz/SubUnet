import torch
from torch import nn

################################################################-Fonctionnement PSPNET:-#############################################################################
# - On applique différents poolings à l'image originale
# - On fait un upsampling du résulat obtenu par chaque pooling
# - On concatène les résultats des upsamplings 
# - On utilise cette nouvelle feature map contenant des informations locales et globales de l'image pour faire notre prédiction avec un "backbone" (segmentation)
################################################################################################################################################################


def single_conv(in_channels, out_channels, pooling=True):
    """(convolution => [BN] => ReLU)"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2) if pooling else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    )


# on définit d'abord le pyramid pooling module 
class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, layer_sizes=[1, 2, 3, 6], block_size=32):
        super(PPM, self).__init__()
        self.features = []
        for layer_size in layer_sizes: # niveaux de poolings du pyramid pooling module
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(layer_size), # le pooling, le parametre est la taille de la sortie
                nn.Conv2d(in_channels, out_channels // len(layer_sizes), kernel_size=3, padding=1), # convolution pour avoir les features maps de chaque pooling
                nn.BatchNorm2d(out_channels // len(layer_sizes)), # on normalise les données
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features) # comme une list mais en mieux 
        self.upsample = nn.Upsample(size=block_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        out = []
        for f in self.features:
            out.append(self.upsample(f(x)))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(PSPNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = nn.Sequential(
            single_conv(self.n_channels, 64, True),
            single_conv(64, 128, True),
            single_conv(128, 256, True)
        )
        self.ppm = PPM(256, 256, layer_sizes=[1, 2, 3, 6])
        self.out_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            single_conv(512, 128, False),
            single_conv(128, 32, False),
            nn.Conv2d(32, self.n_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.in_conv(x)
        xp = self.ppm(x)
        x = torch.cat([x, xp], dim=1)
        x = self.out_conv(x)
        return x
