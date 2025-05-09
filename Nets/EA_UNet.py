import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
from thop import profile
from module.PMDC import PMDC
from module.attention import ChannelAttentionModule

class GhostDV(nn.Module):
    def __init__(self, in_channels, out_channels,raw_planes,cheap_planes):
        super(GhostDV, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.raw_planes = raw_planes
        self.cheap_planes = cheap_planes

        self.conv2 = nn.Conv2d(out_channels+in_channels, raw_planes, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(raw_planes)

        self.conv3 = nn.Conv2d(out_channels+in_channels+raw_planes, raw_planes, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(raw_planes)

        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(raw_planes*2, cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cheap_planes, cheap_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(cheap_planes, cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
        )
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.MSP = MSP(out_channels, out_channels)

    def forward(self, x):
        res_x = self.ch_avg(x)
        input_x = F.relu(self.bn1(self.conv1(x)))

        e = input_x[:,:self.raw_planes]
        c = input_x[:,self.raw_planes:]

        x1 = torch.cat((x,input_x),dim=1)
        x1 = F.relu(self.bn2(self.conv2(x1)))

        x2 = torch.cat((x,input_x,x1),dim=1)
        x2 = F.relu(self.bn3(self.conv3(x2)))

        mix = torch.cat((x1,x2),dim=1)

        m = self.merge(mix)

        c = self.relu(self.cheap(c) + m)

        output = torch.cat((x2,c),dim=1)

        output = self.relu(output + res_x)
        output = self.MSP(output)

        return self.down_sample(output),output


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(in_channels,out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu =nn.ReLU(inplace=True)
        self.CAM = ChannelAttentionModule(in_channels, 16)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = self.CAM(x)
        res_x = self.ch_avg(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.relu(x+res_x)
        return x

class Upsample_block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block1, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = PMDC(in_planes=out_channels, out_planes=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu =nn.ReLU(inplace=True)
        self.CAM = ChannelAttentionModule(in_channels, 16)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = self.CAM(x)
        res_x = self.ch_avg(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.relu(x+res_x)
        return x

class GhostDV1(nn.Module):
    def __init__(self, in_channels, out_channels,raw_planes,cheap_planes):
        super(GhostDV1, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.raw_planes = raw_planes
        self.cheap_planes = cheap_planes

        self.conv2 = nn.Conv2d(out_channels+in_channels, raw_planes, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(raw_planes)

        self.conv3 = nn.Conv2d(out_channels+in_channels+raw_planes, raw_planes, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(raw_planes)

        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(raw_planes*2, cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cheap_planes, cheap_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(cheap_planes, cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.MSP = MSP(out_channels, out_channels)

    def forward(self, x):
        res_x = self.ch_avg(x)
        input_x = F.relu(self.bn1(self.conv1(x)))

        e = input_x[:,:self.raw_planes]
        c = input_x[:,self.raw_planes:]

        x1 = torch.cat((x,input_x),dim=1)
        x1 = F.relu(self.bn2(self.conv2(x1)))

        x2 = torch.cat((x,input_x,x1),dim=1)
        x2 = F.relu(self.bn3(self.conv3(x2)))

        mix = torch.cat((x1,x2),dim=1)

        m = self.merge(mix)

        c = self.relu(self.cheap(c) + m)

        output = torch.cat((x2,c),dim=1)

        output = self.relu(output + res_x)
        output = self.MSP(output)

        return output


class MSP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(MSP, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5,stride=1,padding=5 // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=7, stride=1, padding=7 // 2)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 =nn.Conv2d(4*in_channels,out_channels,kernel_size=1,stride=1,bias=False)
    def forward(self,x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        last =  torch.cat([x,x1,x2,x3],dim=1)
        last = self.relu(self.bn(self.conv1(last)))
        return last

class EA_UNet(nn.Module):
    def __init__(self,args):
        super(EA_UNet, self).__init__()
        self.down_conv1 = GhostDV(in_channels=4,out_channels=64,raw_planes=32,cheap_planes=32)
        self.down_conv2 = GhostDV(in_channels=64, out_channels=128, raw_planes=64, cheap_planes=64)
        self.down_conv3 = GhostDV(in_channels=128, out_channels=256, raw_planes=128, cheap_planes=128)
        self.down_conv4 = GhostDV(in_channels=256, out_channels=512, raw_planes=256, cheap_planes=256)

        self.double_conv = GhostDV1(in_channels=512, out_channels=1024, raw_planes=512, cheap_planes=512)

        self.up_conv4 = Upsample_block(1024, 512)
        self.up_conv3 = Upsample_block(512, 256)
        self.up_conv2 = Upsample_block(256, 128)
        self.up_conv1 = Upsample_block1(128, 64)

        self.conv_last = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)

        x = self.double_conv(x)

        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)

        x = self.conv_last(x)

        return x

