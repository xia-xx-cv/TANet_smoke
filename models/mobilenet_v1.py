import torch, math
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .sub_modules import SELayer_Me

def conv_bn(inp, oup, stride, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm(oup),
        nn.ReLU6(inplace=True)
    )


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class BaseNet(nn.Module):
    def __init__(self, pretrained=True, bn=None):
        super(BaseNet, self).__init__()
        # copying modules from pretrained models
        self.pretrained = create_model(pretrained=pretrained, bn=bn)

    def base_forward(self, x):
        low = self.pretrained.low_level_features(x)
        mid = self.pretrained.mid_level_features(low)
        x = self.pretrained.high_level_features(mid)
        return low, mid, x


def create_model(pretrained=True, output_stride=8,
                 width_mult=1., bn=nn.BatchNorm2d):
    model = MobileNetV2(pretrained=pretrained,
                        output_stride=output_stride,
                        width_mult=width_mult,
                        bn=bn)
    return model


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, dilation=1, expand_ratio=1,
                 BatchNorm=nn.BatchNorm2d, att=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation
        self.att = att
        if att:
            self.att = SELayer_Me(oup)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, 1, bias=False),
                BatchNorm(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, 1, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, bias=False),
                BatchNorm(oup),
            )

    def forward(self, x):
        x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
        out = self.conv(x_pad)
        if self.att:
            out = self.att(out)
        if self.use_res_connect:
            x = x + out
        else:
            x = out
        return x


class MobileNetV2(nn.Module):
    def __init__(self, output_stride=8, bn=nn.BatchNorm2d,
                 width_mult=1., pretrained=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [
        # ---- t, c, n, s ----
            # [1, 16, 1, 1],
            # [6, 24, 2, 2],  # low level feat
            # [6, 32, 3, 1],
            # [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
            [1, 16, 1, 1, True],
            [6, 32, 2, 2, True],  # low level feat
            [6, 64, 3, 2, False],
            [6, 96, 2, 1, False],  # mid level feat
            [6, 160, 2, 1, False],
            [6, 160, 1, 1, False],  # high level feat
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.features = [conv_bn(inp=3, oup=input_channel, stride=2, BatchNorm=bn)]
        current_stride *= 2
        # building inverted residual blocks
        for t, c, n, s, att in interverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel,
                                               stride, dilation, t, bn, att))
                else:
                    self.features.append(block(input_channel, output_channel,
                                               1, dilation, t, bn, att))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

        if pretrained:
            self._load_pretrained_model()
        self.low_level_features = self.features[0:4]  # 24x64x64  /4
        self.mid_level_features = self.features[4:9]  # 32x32x32  /2
        self.high_level_features = self.features[9:]  # 320x16x16  /2

    def forward(self, x):
        # 32x64x64, 96x64x64, 160x64x64
        low_feat = self.low_level_features(x)
        mid_feat = self.mid_level_features(low_feat)
        x = self.high_level_features(mid_feat)
        return low_feat, mid_feat, x

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict, strict=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    input = torch.rand(1, 3, 256, 256)
    model = MobileNetV2(output_stride=16, bn=nn.BatchNorm2d)
    low, mid, high = model(input)
    print(low.size(), mid.shape, high.shape)
