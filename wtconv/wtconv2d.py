import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .util import wavelet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Create_lowpass_filter(size):
    return torch.ones(size, size) / (size * size)


def apply_lowpass_filter(image, kernel, channel):
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channel, channel, kernel.shape[2], kernel.shape[3])
    filter_image = F.conv2d(image, kernel.to(device), padding=(kernel.shape[2] // 2))
    return filter_image


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1
        self.kernel_size = kernel_size

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape

            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # curr_x_ll.shape  torch.Size([1, 48, 24, 24])
            curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]  # 第一次分解之后的低频图像

            shape_x = curr_x.shape  # shape_x: torch.Size([1, 48, 4, 12, 12])

            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll  # curr_x_ll shape: torch.Size([1, 48, 12, 12])

            # 我们需要对这个高频部分进行平滑处理，从而减少异常影响   ########
            curr_h_shape = curr_x_h.shape
            reshape_curr_x_h = curr_x_h.reshape(curr_h_shape[0], curr_h_shape[1] * 3, curr_h_shape[3], curr_h_shape[4])
            kernel = Create_lowpass_filter(size=self.kernel_size)
            filter_reshape_curr_x_h = apply_lowpass_filter(reshape_curr_x_h, kernel, channel=curr_h_shape[1] * 3)
            filterd_hf = filter_reshape_curr_x_h.reshape(curr_h_shape[0], curr_h_shape[1], 3,
                                                         curr_h_shape[3], curr_h_shape[4])

            # 将对应尺度的低频和高频合并，然后进行逆变换
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), filterd_hf], dim=2)
            next_x_ll = wavelet.inverse_wavelet_transform(curr_x, self.iwt_filter)
            # 由于填充影响，将多余部分剪切掉
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # x = self.base_scale(self.base_conv(x))
        # x = x + x_tag
        x = x_tag
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


if __name__ == "__main__":

    f1 = torch.randn(1, 48, 23, 23)
    f2 = torch.randn(1, 48, 23, 23)
    model = WTConv2d(in_channels=48, out_channels=48, kernel_size=1)
    out = model(f1)
    print('out shape:', out.shape)
