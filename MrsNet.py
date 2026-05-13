import torch
import torch.nn as nn
from Spa_Fre_Fuse_Module import Spa_Fre_Fuse_Module


class MrsNet(nn.Module):

    def __init__(self, in_ch=189, out_ch=189, head_ch=32):
        super().__init__()
        n1 = int(in_ch / 2)

        # 设计浅层特征映射块，将输入 HSI 映射到特征空间
        self.Encoder1 = nn.Sequential(nn.Conv2d(in_ch, n1, kernel_size=3, padding=1, padding_mode='reflect'),
                                      nn.BatchNorm2d(n1),
                                      nn.LeakyReLU())

        self.Encoder2 = nn.Sequential(nn.Conv2d(n1, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                      nn.BatchNorm2d(head_ch),
                                      nn.Sigmoid())

        # 调制网络
        self.FC1 = nn.Sequential(nn.Conv2d(in_ch, n1, kernel_size=1),
                                 nn.BatchNorm2d(n1),
                                 nn.Sigmoid())

        self.FC2 = nn.Sequential(nn.Conv2d(n1, head_ch, kernel_size=1),
                                 nn.BatchNorm2d(head_ch),
                                 nn.Sigmoid())

        # 双域多尺度特征提取
        self.Spa_Fre_Fuse1 = Spa_Fre_Fuse_Module(head_ch=head_ch)
        self.Spa_Fre_Fuse2 = Spa_Fre_Fuse_Module(head_ch=head_ch)
        self.Spa_Fre_Fuse3 = Spa_Fre_Fuse_Module(head_ch=head_ch)

        # 解码网络，恢复背景
        self.Decoder1 = nn.Sequential(nn.Conv2d(head_ch, n1, kernel_size=3, padding=1, padding_mode='reflect'),
                                      nn.BatchNorm2d(n1),
                                      nn.LeakyReLU())

        self.Decoder2 = nn.Sequential(nn.Conv2d(n1, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                      nn.BatchNorm2d(out_ch), )

    def forward(self, x, center):
        x1 = self.Encoder1(x)
        x2 = self.Encoder2(x1)

        fuse_features_1 = self.Spa_Fre_Fuse1(x2)
        fuse_features = self.Spa_Fre_Fuse2(fuse_features_1)

        fc1_out = self.FC1(center)
        fc2_out = self.FC2(fc1_out)

        out1 = self.Decoder1(fuse_features + fuse_features * fc2_out + x2)
        out2 = self.Decoder2(out1 + out1 * fc1_out + x1)

        return out2
