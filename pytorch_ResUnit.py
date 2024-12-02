
import torch
import torch.nn as nn


class TorchResUnit(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, if_down_sample=False, device = "cuda"):
        self.device = device
        super(TorchResUnit, self).__init__()
        if_down_sample = int(if_down_sample)
        self.norm = nn.LayerNorm(img_size//(if_down_sample+1),device=self.device)
        self.if_down_sample = if_down_sample
        self.conv2d_01 = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    dilation=1,
                                    groups=1,
                                    bias=True,
                                    padding_mode='zeros',
                                         device=self.device)
        self.act_func1 = nn.LeakyReLU(0.1)

        self.conv2d_02 = torch.nn.Conv2d(out_channels,
                                        out_channels,
                                    1,
                                    stride=1,
                                    padding=0,
                                    dilation=1,
                                    groups=1,
                                    bias=True,
                                    padding_mode='zeros',
                                         device=self.device)
        self.act_func2 = nn.LeakyReLU(0.1)

        self.conv2d_03 = torch.nn.Conv2d(out_channels,
                                         in_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    dilation=1,
                                    groups=1,
                                    bias=True,
                                    padding_mode='zeros',
                                         device=self.device)
        self.act_func3 = nn.LeakyReLU(0.1)

        self.conv2d_down = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                    3,
                                    stride=2,
                                    padding=1,
                                    dilation=1,
                                    groups=1,
                                    bias=True,
                                    padding_mode='zeros',
                                           device=self.device)
        self.act_func_down = nn.LeakyReLU(0.1)
        self.conv2d_notdown = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    dilation=1,
                                    groups=1,
                                    bias=True,
                                    padding_mode='zeros'
                                              ,device=self.device)
        self.act_func_notdown = nn.LeakyReLU(0.1)

    def forward(self,x):
        y = self.act_func1(self.conv2d_01(x))

        y = self.act_func2(self.conv2d_02(y))
        y = self.act_func3(self.conv2d_03(y))
        if self.if_down_sample:
            y = self.norm(self.act_func_down(self.conv2d_down(x+y)))
        else:
            y = self.norm(self.act_func_notdown(self.conv2d_notdown(x+y)))
        return y


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# XX = torch.randn(size=[3,16,128,128]).to(device)
# TorchResUnit_01 = TorchResUnit(128,16,8,False)
# RESS = TorchResUnit_01.forward(XX)
#
# print(RESS.shape)
#
# class HoleConv2d(nn.Module):
#     pass




