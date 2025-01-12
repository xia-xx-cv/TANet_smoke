import math, torch
import torch.nn as nn
# from torch.nn.functional import interpolate
import torch.nn.functional as F


class LinMe(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, active=True):
        super(LinMe, self).__init__()
        if active:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                          kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU())
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                          kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(out_ch))
    #     self._init_weight()
    #
    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

    def forward(self, x):
        x = self.layer(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ConcurrentModule(nn.ModuleList):
    r""" The outputs of the layers are concatenated at channel dimension.
    [out1, out2, out3....] """
    def __init__(self, modules=None):
        super(ConcurrentModule, self).__init__(modules)

    def forward(self, x):
        outputs = []
        for layer in self:
            outputs.append(layer(x))
        return torch.cat(outputs, 1)

# texture-aware (TA) module
class _OrientModule(nn.Module):
    def __init__(self, inplanes, planes, kh, kw, padding=0, dilation=1, bn=nn.BatchNorm2d):
        super(_OrientModule, self).__init__()
        self.atrous_w = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=(kh, 1),
                                  stride=1, padding=padding,
                                  dilation=dilation, bias=False),
            bn(planes),
            nn.ReLU6(),
        )

        self.atrous_h = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=(1, kw),
                                  stride=1, padding=padding,
                                  dilation=dilation, bias=False),
            bn(planes),
            nn.ReLU6(),
        )
        # self.ch_att_h = ChannelAttention(planes)

        self.fuse1 = nn.Conv2d(planes, inplanes, 3, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU6()
        self.c = planes

    def EPS(self, B, H, W):
        # return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
        return (torch.eye(H, W)*(1e-10)).cuda().unsqueeze(0).repeat(B, 1, 1)

    def forward(self, x):
        b, _, h, w = x.size()
        xw = self.atrous_w(x)  # b c 1 w
        xh = self.atrous_h(x)  # b c h 1
        # b c h 1--> b 1 c h--> b c h--> b h c
        xh_t = xh.permute(0, 3, 1, 2).contiguous().view(b * 1, -1, h).permute(0, 2, 1)
        # b c 1 w--> b 1 c w--> b c w
        xw_t = xw.permute(0, 2, 1, 3).contiguous().view(b * 1, -1, w)
        # b h w --> b 1 h w
        xhw_att = torch.bmm(xh_t, xw_t) + self.EPS(b, h, w)
        xhw_att = xhw_att.unsqueeze(1)
        # xhw = self.conv2_3(x) + self.conv2_4(x)
        xhw = self.fuse1(F.relu(F.interpolate(xh, (h, w), mode='bilinear', align_corners=True)
                                + F.interpolate(xw, (h, w), mode='bilinear', align_corners=True)
                                ))

        xhw = xhw + x
        return xhw_att.sigmoid(), xhw


class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # same as mxnet.symbol.Reshape()
        nn.init.kaiming_uniform_(self.conv.weight)  # initializing conv only when creating the DUC object

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class SELayer_Me(nn.Module):
    def __init__(self, planes, reduction=64):
        super(SELayer_Me, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, planes),
            # nn.Sigmoid(),
        )
        self.activate = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.pool1(x).view(b, c)
        y1 = self.fc(y1).view(b, c, 1, 1)
        y2 = self.pool2(x).view(b, c)
        y2 = self.fc(y2).view(b, c, 1, 1)
        return x * self.activate(y1 + y2)


class SELayer(nn.Module):
    def __init__(self, planes, reduction=64):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SepConvLayer(nn.Module):
    def __init__(self, inplanes, planes, k_size, cuda=True):
        super(SepConvLayer, self).__init__()
        self.dw_conv = list()
        for _ in range(inplanes):
            self.dw_conv.append(nn.Conv2d(1, 1, kernel_size=k_size,
                                          padding=k_size//2,
                                          bias=False).cuda())
        self.pw_conv = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU6(inplace=True)
        self.conv = nn.Conv2d(planes, 1, 1, bias=False)

    def forward(self, x):
        ch = x.shape[1]
        for i in range(ch):
            x[:, i:] = self.relu(self.dw_conv[i](x[:, i:i+1]))
        x = self.pw_conv(x)
        x = self.relu(self.bn(x))
        x = self.conv(x)
        return x


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        # self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def INF(self, B, H, W):
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
        # return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().\
            view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().\
            view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) +
                    self.INF(m_batchsize, height, width)).\
            view(m_batchsize, width, height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat((energy_H, energy_W), 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().\
            view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().\
            view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).\
            view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).\
            view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class RCCAModule(nn.Module):
    def __init__(self, inplanes, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        planes = inplanes // 4
        # planes = 128
        self.conva = nn.Sequential(nn.Conv2d(inplanes, planes, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.LeakyReLU(True))
        self.cca = CrissCrossAttention(planes)
        self.convb = nn.Sequential(nn.Conv2d(planes, planes, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.LeakyReLU(True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes+planes, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=True)
            )
        # self.cls = nn.Conv2d(out_channels, num_classes, kernel_size=3, stride=2, bias=True)

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat((x, output), 1))
        return output



