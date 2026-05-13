import torch
import torch.nn as nn


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class MultiScale_Spatial_Module(nn.Module):
    def __init__(self, head_ch=32):
        super().__init__()

        # body
        self.Conv1x1 = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=1),
                                     nn.BatchNorm2d(head_ch),
                                     nn.LeakyReLU(inplace=True))

        self.Conv3x3 = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                     nn.BatchNorm2d(head_ch),
                                     nn.LeakyReLU())

        self.Conv5x5 = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=5, padding=2, padding_mode='reflect'),
                                     nn.BatchNorm2d(head_ch),
                                     nn.LeakyReLU(inplace=True))

        self.fuse = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                  nn.BatchNorm2d(head_ch),
                                  nn.LeakyReLU())

        self.mask_conv = CentralMaskedConv2d(head_ch, head_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x2 = self.mask_conv(x)
        s1 = self.Conv1x1(x2)
        s2 = self.Conv3x3(x2)
        s3 = self.Conv5x5(x2)
        muti_scale = self.fuse(s1 + s2 + s3) + x2

        return muti_scale

