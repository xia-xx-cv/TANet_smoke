from __future__ import division
import torch.nn as nn
from math import sqrt
from torch.nn.functional import interpolate
import torch
# from .base_me import BaseNet
from models.mobilenet_v1 import MobileNetV2
from models.mobilenet_v1 import InvertedResidual as ResBlock
from sub_modules import LinMe, DUC, RCCAModule, _OrientModule, \
    SpatialAttention, ChannelAttention

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class MyNet(nn.Module):
    def __init__(self, nclass, backbone, pretrained=False,
                 bn=nn.BatchNorm2d, map_size=(256, 256)):
        super(MyNet, self).__init__()
        if backbone == 'resnet50' or backbone == 'resnet101':
            low_ch, x_ch, deep = 256, 2048, True
        elif backbone == '18' or backbone == '34':
            low_ch, x_ch, deep = 64, 512, False
        else:  # mobilenet--paper
            low_ch, x_ch, deep = 24, 64, False
        # MobileNetV2 with attention modules in low level feature layers
        self.backbone = MobileNetV2(output_stride=16, bn=bn, pretrained=pretrained)
        self.head = RCCAModule(160, 96, 32)  # 96
        self.auxlayer = AuxLayer(24, x_ch, bn, map_size)  # 32x32x32
        self.decoder = Decoder(low_ch, nclass, deep)  # 24x64-, 320x16-

    def forward(self, x, y=None, th=0.2):
        _, _, h, w = x.size()
        x = interpolate(x, ((h // 2) * 2, (w // 2) * 2), **up_kwargs)
        low_fea, mid_fea, x = self.backbone(x)
        # low, mid, x: 32x64-, 96x32- , 160x32-
        x = self.head(x)  # 96x32-
        # # --2020.08.11----
        ''' aux path: extracting attentioned features from mid-and high- level features'''
        aspp, orient = self.auxlayer(interpolate(mid_fea, (h//4, w//4), **up_kwargs))  # 96x64x64
        if self.training:
            out, aux = self.decoder(low_fea, aspp, orient, x)  # 32x64-  96x64-  96x32
            aux = interpolate(aux.float(), (h, w), **up_kwargs)
            # aux = interpolate(aux, (h, w), **up_kwargs)
            out = interpolate(out, (h, w), **up_kwargs)
            return out.sigmoid(), aux
        else:
            out = self.decoder(low_fea, aspp, orient, x)
            out = interpolate(out, (h, w), **up_kwargs)
            return out.sigmoid()


class Decoder(nn.Module):
    def __init__(self, in_chs, out_chs, deep=True):
        super(Decoder, self).__init__()
        # inter_ch = 64
        self.deep = deep
        self.out_chs = out_chs
        ch = 64
        # self.cls = ChannelAttention(out_chs)

        self.fuse1 = nn.Sequential(
            ResBlock(ch*2, ch*2),
            LinMe(ch*2, ch//8, k=1),
        )

        self.duc = DUC(ch//2, ch//2)  # 32x64x64 -> 8x128x128
        self.fuse2 = nn.Sequential(
            ResBlock(ch//8, ch//8),
            LinMe(ch//8, out_chs+2, active=False)
        )

        if self.training:
            self.fuse = nn.Sequential(
                ResBlock(ch*2, ch*2),
                LinMe(ch*2, out_chs, k=1, active=False),
            )
            # self.out = nn.Sequential(
            #     ResBlock(ch * 2, ch * 2),
            #     LinMe(ch*2, out_chs-1, k=1, active=False),
            # )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, low_fea, aspp, orient, x):
        # low: 8, 32, 64, 64
        # mid--aspp: 8, 512, 64, 64
        # mid--orient: 8, 1024, 64, 64
        # x: 8, 160, 32, 32
        _, _, h, w = low_fea.size()
        x = interpolate(x, (h, w), **up_kwargs)  # 32
        if x.size() != orient.size():
            x = interpolate(x, orient.size()[2:], **up_kwargs)
        out1 = interpolate(torch.cat((orient, x), dim=1), (h*2, w*2), **up_kwargs)
        out1 = self.fuse1(out1)
        out1 = self.fuse2(self.duc(low_fea) + out1)
        if self.training:
            out2 = torch.cat((low_fea, aspp), dim=1)
            # out2 = self.out(out2)
            # return out1, out2
            out2 = self.fuse(out2)
            return out1, out2.softmax(1).max(1)[-1].unsqueeze(1)
        else:
            return out1


class AuxLayer(nn.Module):
    def __init__(self, in_chs, out_chs, bn, map_size):
        super(AuxLayer, self).__init__()
        self.midhead = build_aspp('mobilenet', 16, bn)
        # self.activate = nn.ReLU6()
        h, w = map_size
        h, w = h//4, w//4
        self.att = _OrientModule(96, in_chs, kw=w, kh=h)
        self.relu = nn.ReLU
        self.size = (h, w)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, mid):
        ori_size = mid.size()[2:]
        _, _, _, mid = self.midhead(mid)  # 96x64x64
        if ori_size != self.size:
            att, orient = self.att(interpolate(mid, self.size, **up_kwargs))
            return mid, interpolate(orient*att, ori_size, **up_kwargs)
        else:
            att, orient = self.att(mid)
            return mid, att*orient


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, bn):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, bn):
        super(ASPP, self).__init__()
        inplanes = 96
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        # k=1 3 3 3, d=1 6 12 18,  win= (d-1)*(k-1)+k = 1 13 25 37
        # 1 12 24 36 --> 1 25 49 73
        ch = inplanes
        # ch2 = 256
        self.aspp1 = _ASPPModule(inplanes, ch, 1, padding=0, dilation=dilations[0], bn=bn)
        self.aspp2 = _ASPPModule(inplanes, ch, 3, padding=dilations[1], dilation=dilations[1], bn=bn)
        self.aspp3 = _ASPPModule(inplanes, ch, 3, padding=dilations[2], dilation=dilations[2], bn=bn)
        self.aspp4 = _ASPPModule(inplanes, ch, 3, padding=dilations[3], dilation=dilations[3], bn=bn)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, ch, 1, stride=1, bias=False),
                                             bn(ch),
                                             nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch*5, inplanes, 1, bias=False),
            bn(inplanes),
            nn.ReLU6(True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        # res = x
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = interpolate(x5, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        return x2, x3, x4, x


def build_aspp(backbone, output_stride, bn):
    return ASPP(backbone, output_stride, bn)

