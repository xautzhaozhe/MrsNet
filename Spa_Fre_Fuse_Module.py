import torch
import torch.nn as nn
from MultiScale_Spatial_Module import MultiScale_Spatial_Module
from MultScale_Wavelet import MultiScale_Wavelet_Module


class Spa_Fre_Fuse_Module(nn.Module):
    def __init__(self, head_ch=32):
        super().__init__()

        self.Spa_branch = MultiScale_Spatial_Module(head_ch=head_ch)
        self.Fre_branch = MultiScale_Wavelet_Module(head_ch=head_ch)

        self.Conv3x3 = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                     nn.BatchNorm2d(head_ch),
                                     nn.LeakyReLU())

        self.avgpool_1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_2 = nn.AdaptiveAvgPool2d(1)
        self.Sigmoid = nn.Sigmoid()

        self.fuse = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                  nn.BatchNorm2d(head_ch),
                                  nn.LeakyReLU())

    def forward(self, x):
        Spa = self.Spa_branch(x)
        fre = self.Fre_branch(x)

        Spa_weight = self.Sigmoid(self.avgpool_1(self.Conv3x3(Spa)))
        Fre_weight = self.Sigmoid(self.avgpool_2(self.Conv3x3(fre)))
        Spa_out = Spa + Spa_weight*fre
        fre_out = fre + Fre_weight*Spa
        fuse_features = self.fuse(Spa_out+fre_out) + x

        return fuse_features

