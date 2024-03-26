import numpy as np
import os
import torch
from sklearn.metrics import roc_auc_score


class AvgMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0.0, 0.0, 0.0, 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_cls_metrics(pred, GT, th=0.2):
    output = torch.where(pred >= th, torch.ones_like(pred), torch.zeros_like(pred))
    target = torch.where(GT >= th, torch.ones_like(GT), torch.zeros_like(GT))
    histmap = output + target * 2
    tn, fp, fn, tp = torch.histc(histmap.cpu(), bins=4, min=0, max=3)
    num = target.numel()
    # assert num - tn == (tp + fp + fn)
    iou = tp / (tp + fp + fn)
    # self.iou[torch.isnan(self.iou)] = 0

def cal_mse(output, gt):
    sq_err = (output - gt)**2
    h, w = gt.size()[2:]
    mse = sq_err.sum()/(h*w)
    # return mse


class AuxTrain(object):
    """ modified based on 'semseg' codes
        2020.08--xia """
    def __init__(self, base_lr):
        self.base_lr = base_lr
        self.lr = base_lr

    def step_lr(self, epoch, step_epoch, multiplier=0.1):
        """Sets the learning rate to the base LR decayed by 10 every step epochs"""
        self.lr = self.base_lr * (multiplier ** (epoch // step_epoch))
        return self.lr

    def poly_lr(self, curr_iter, max_iter, power=0.9):
        """poly learning rate policy"""
        self.lr = self.base_lr * (1 - float(curr_iter) / max_iter) ** power
        return self.lr


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
