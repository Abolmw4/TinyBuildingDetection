from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable

# def conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool=True) -> nn.Conv2d:
#     return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

def conv(**kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels=kwargs.get('in_channels'), out_channels=kwargs.get('out_channels'), kernel_size=kwargs.get('kernel_size'), padding=(kwargs.get('kernel_size')//2), bias=kwargs.get('bias', True))

class Upsampler(nn.Sequential):
    def __init__(self, conv_layer: Callable[..., nn.Conv2d], scale: int, number_of_feature: int, bn: bool=False, act: bool=False, bias: bool=True) -> None:

        m = []
        if (scale & (scale - 1)) == 0:    # It means scale is a power of two (e.g., 1, 2, 4, 8, 16, etc.)
            for _ in range(int(math.log(scale, 2))):
                m.append(conv_layer(in_channels=number_of_feature, out_channels=4 * number_of_feature, kernel_size=3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(number_of_feature))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv_layer(in_channels=number_of_feature, out_channels=9 * number_of_feature, kernel_size=3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(number_of_feature))
            if act: m.append(act())
        else:
            raise NotImplementedError
        super().__init__(*m)

class ResBlock(nn.Module):
    def __init__(self, number_of_features: int, kernel_size: int, bias: bool=True, bn: bool=False, act: nn=nn.ReLU(True), res_scale: int=1) -> None:
        super().__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(in_channels=number_of_features, out_channels=number_of_features, kernel_size=kernel_size, bias=bias, padding=(kernel_size//2)))
            if bn: m.append(nn.BatchNorm2d(number_of_features))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x) -> Tensor:
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Decoder(nn.Module):
    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(c1, c1 // 2, 1, bias=False)
        self.conv2 = nn.Conv2d(c2, c2 // 2, 1, bias=False)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d((c1 + c2) // 2, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 64, kernel_size=1, stride=1))
        self.__init_weight()

    def forward(self, low_level_feature: Tensor, high_level_feature: Tensor, scaler_factor: int) -> Tensor:
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)

        high_level_feature = self.conv2(high_level_feature)
        high_level_feature = self.relu(high_level_feature)
        high_level_feature = F.interpolate(high_level_feature, size=[i*(scaler_factor//2) for i in low_level_feature.size()[2:]], mode='bilinear', align_corners=True)
        if scaler_factor>1:
            low_level_feature = F.interpolate(low_level_feature, size=[i*(scaler_factor//2) for i in low_level_feature.size()[2:]], mode='bilinear', align_corners=True)
        high_level_feature = torch.cat((high_level_feature, low_level_feature), dim=1)
        high_level_feature = self.last_conv(high_level_feature)

        return high_level_feature


    def __init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EDSR(nn.Module):
    def __init__(self, num_channels: int=3, input_channel: int=64, scale_factor: int=4, width: int=64, number_resblock: int=16, kernel_size: int=3) -> None:
        super().__init__()
        self.head = nn.Sequential(*[nn.Conv2d(input_channel, width, kernel_size, padding=(kernel_size//2), bias=True)])
        self.body = nn.Sequential(*[ResBlock(number_of_features=width, kernel_size=kernel_size, act=nn.ReLU(), res_scale=1) for _ in range(number_resblock)],
                                  nn.Conv2d(in_channels=width, out_channels=width, kernel_size=kernel_size, padding=(kernel_size//2), bias=True))
        self.tail = nn.Sequential(*[Upsampler(conv_layer=conv, scale=scale_factor, number_of_feature=width, act=False), conv(in_channels=width, out_channels=num_channels, kernel_size=kernel_size, bias=True)])

    def forward(self, x) -> Tensor:
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


class SuperResolution(nn.Module):
    def __init__(self, num_channel: int, c1: int, c2: int, scale_factor: int) -> None:
        '''

        :param num_channel: this is for out channel of EDSR block
        :param c1: input channel for decoder
        :param c2: output channel for decoder
        :param scale_factor: magnitude factor
        '''
        super().__init__()
        self.decoder = Decoder(c1=c1, c2=c2)
        self.edsr = EDSR(num_channels=num_channel, input_channel=64, scale_factor=8)
        self.scale_factor = scale_factor

    def forward(self, low_level_feature: torch.tensor, high_level_feature: torch.tensor) -> Tensor:
        x_sr= self.decoder(high_level_feature, low_level_feature, self.factor)
        x_sr_up = self.edsr(x_sr)
        return x_sr_up

