import torch
import torch.nn as nn
from wtconv.wtconv2d import WTConv2d


class MultiScale_Wavelet_Module(nn.Module):
    def __init__(self, head_ch=32):
        super().__init__()

        # body
        self.Conv3x3 = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                     nn.BatchNorm2d(head_ch),
                                     nn.LeakyReLU())
        self.WTConv1x1 = WTConv2d(in_channels=head_ch, out_channels=head_ch, kernel_size=1)

        self.WTConv3x3 = WTConv2d(in_channels=head_ch, out_channels=head_ch, kernel_size=3)

        self.WTConv5x5 = WTConv2d(in_channels=head_ch, out_channels=head_ch, kernel_size=5)

        self.fuse = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                  nn.BatchNorm2d(head_ch),
                                  nn.LeakyReLU())

    def forward(self, x):

        x1 = self.Conv3x3(x)
        s1 = self.WTConv1x1(x1)
        s2 = self.WTConv3x3(x1)
        s3 = self.WTConv5x5(x1)
        WT_muti_scale = self.fuse(s1 + s2 + s3) + x

        return WT_muti_scale








