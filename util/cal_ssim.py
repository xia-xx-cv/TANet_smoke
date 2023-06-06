import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(self.window_size, self.channel)

    def gaussian(self, window_size, sigma):
        para = [exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)]
        gauss = torch.Tensor(para)
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            channel, 1,
            window_size, window_size).contiguous())
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        self.window = window
        self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class MSE_bin(nn.Module):
    """ Computing mse for binary maps
        2021.01--xia """
    def __init__(self, th=0.2):
        super(MSE_bin, self).__init__()
        self.th = th
        self.criterion = nn.MSELoss()
        # self.batch_size = batch_size

    def forward(self, x, GT):
        # x = torch.where(x >= self.th, torch.ones_like(x), torch.zeros_like(x))
        target = torch.where(GT >= self.th, torch.ones_like(GT), torch.zeros_like(GT))
        if GT.size()[2:] != x.size()[2:]:
            target = F.interpolate(GT, x.shape[2:],
                                   mode='bilinear', align_corners=True)
        return self.criterion(x, target)


class IoU_Loss(nn.Module):
    """ Computing iou for binary maps
        2021.01--xia """
    def __init__(self, th=0.2):
        super(IoU_Loss, self).__init__()
        self.th = th
        self.criterion = nn.MSELoss()
        # self.zero_grad()
        # self.batch_size = batch_size

    def forward(self, x, GT):
        # bs, c = x.shape[0], x.shape[1]
        x = torch.where(x >= self.th, torch.ones_like(x), torch.zeros_like(x))
        target = torch.where(GT >= self.th, torch.ones_like(GT), torch.zeros_like(GT))
        if target.size()[2:] != x.size()[2:]:
            target = F.interpolate(target, x.shape[2:], mode='nearest')

        # calculating iou batch-wise
        tp = ((x*target)==1).sum(3).sum(2).sum(1)
        union = ((x+target)!=0).sum(3).sum(2).sum(1)
        iou = tp.float()/union.float()
        iou[torch.isnan(iou)] = 0.0  # iou size: bx1
        return torch.mean(iou)


if __name__ == "__main__":
    img = torch.rand(1, 1, 20, 20)
    img2 = torch.rand(1, 1, 20, 20)
    print(_ssim(img, img2))
