import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from msgfusion.models.fusion_operators import L1Fusion, addition_fusion


class ConvLayer(torch.nn.Module):
    """Convolution + reflect padding; ReLU everywhere except the terminal layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out


class DenseConv2d(torch.nn.Module):
    """DenseNet-style coupling: concatenate input with a ConvLayer branch."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(torch.nn.Module):
    """Three consecutive DenseConv2d stages; width schedule matches released weights."""

    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [
            DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
            DenseConv2d(in_channels + out_channels_def, out_channels_def, kernel_size, stride),
            DenseConv2d(in_channels + out_channels_def * 2, out_channels_def, kernel_size, stride),
        ]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class DenseFuseNet(nn.Module):
    """DenseFuse skeleton: encoder used in MSGFusion for attention priors; decoder kept."""

    def __init__(self, input_nc=1, output_nc=1):
        super(DenseFuseNet, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

    def encoder(self, input_tensor):
        x1 = self.conv1(input_tensor)
        x_DB = self.DB1(x1)
        return [x_DB]

    def fusion(self, en1, en2, strategy_type='addition'):
        if strategy_type is 'L1Fusion':
            merge = L1Fusion
        else:
            merge = addition_fusion

        f_0 = merge(en1[0], en2[0])
        return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return [output]


DenseFuse_net = DenseFuseNet
