import os, time, argparse
# import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torchvision.utils import save_image
from datasets import data_reader_syn as data_reader
# from util import transform_me as trans
from util import transform as trans
from util import config
from util.saver import Saver
from tensorboardX import SummaryWriter
from util.util_smoke import AvgMeter, Metrics, AuxTrain
from models.mynet_0205_ori import MyNet
from util.cal_ssim import SSIM, MSE_bin
import numpy as np
import random


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_parser():
    parser = argparse.ArgumentParser(description='Dual smoke density estimation')
    parser.add_argument('--config', type=str, default='smoke_ori.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def main():
    global args, writer, saver
    args = get_parser()

    seed = args.seed
    print(seed)
    # seed_all(seed)
    
    # cudnn.benchmark, cudnn.deterministic = True, True
    criterion = nn.MSELoss()
    # criterion2 = MSE_bin(th=0.2)
    criterion2 = SSIM()
    if args.arch.lower() == 'tanet':
        model = MyNet(nclass=args.classes, backbone=args.backbone,
                      pretrained=False, bn=nn.BatchNorm2d)
        modules_ori = []
        modules_new = [model.backbone, model.head, model.auxlayer, model.decoder]  # smoke
    else:
        raise RuntimeError("=> Unknown network architecture: {}".format(args.arch))
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    args.index_split = 1
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    saver = Saver(args)
    saver.save_experiment_config()
    writer = SummaryWriter(log_dir=os.path.join(saver.experiment_dir))
    # writer = TensorboardSummary(saver.experiment_dir).create_summary()

    # logger.info(args), logger.info(model)
    if args.train_gpu:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
    mean, std = [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]
    tr_trans = trans.Compose([
        trans.RandomRotation([args.rotate_min, args.rotate_max]),
        trans.RandomHorizontalFlip(),  # p=50%
        trans.RandomVerticalFlip(),
        # --- scale(ratio change-probability =50%) and crop
        trans.RandomResizedCrop(scale=[args.scale_min, args.scale_max], size=args.train_size),
        trans.ToTensor(),
        trans.Normalize(mean=mean, std=std)
    ])

    train_data = data_reader.Smoke(split='train', data_root=args.data_root,
                                   data_list=args.train_list, transform=tr_trans)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True,
                                               sampler=None, drop_last=True)

    if args.evaluate:
        val_trans = trans.Compose([
            trans.Resize(),
            trans.ToTensor(),
            trans.Normalize(mean=mean, std=std)
        ])
        # val_data = data_reader.Smoke(split='val', data_root=args.data_root,
        #                              data_list=args.val_list, transform=val_trans, synth=False)
        val_data = data_reader.Smoke(split='val', data_root=args.data_root,
                                     data_list=args.val_list, transform=val_trans)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                 shuffle=False, num_workers=args.workers,
                                                 pin_memory=True, sampler=None)

    loss_tr = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1  # counting from 1
        f_loss_tr, mSSIM_tri, mIoU_tr = train(train_loader, model, optimizer, epoch, criterion, criterion2)
        loss_tr += f_loss_tr
        writer.add_scalar('train/loss', f_loss_tr, epoch_log)
        writer.add_scalar('train/mIoU', mIoU_tr, epoch_log)
        # writer.add_scalar('mAcc_train', mAuc_train, epoch_log)

        if epoch_log == 1:
            filename = saver.experiment_dir + '/train_epoch_' + str(epoch_log) + '.pth'
            print('Saving checkpoint to: ' + filename)
            torch.save(model, filename)
            print(loss_tr / epoch_log)
        elif epoch_log % args.save_freq == 0:
            filename = saver.experiment_dir + '/train_epoch_' + str(epoch_log) + '.pth'
            print('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, filename)
            print(loss_tr/epoch_log)


        if args.evaluate:
            loss_val, mIoU_val, mSSIM = validate(epoch_log, val_loader,
                                                 model, criterion, criterion2)
            writer.add_scalar('val/loss', loss_val, epoch_log)
            writer.add_scalar('val/mIoU', mIoU_val, epoch_log)
            writer.add_scalar('val/mSSIM', mSSIM, epoch_log)
            # writer.add_scalar('val/allAcc', allAcc_val, epoch_log)


def train(train_loader, model, optimizer, epoch, criterion, criterion2):
    # kwargs = {'mode': 'bilinear', 'align_corners': True}
    batch_time, data_time = AvgMeter(), AvgMeter()
    main_loss_meter, loss_meter = AvgMeter(), AvgMeter()
    l_tr, iou_meter, ss = 0.0, AvgMeter(), 0.0
    count = 0.0

    model.train()
    # tbar = tqdm(train_loader, position=0, leave=True)
    num_img_tr = len(train_loader)
    max_iter = args.epochs * num_img_tr
    end = time.time()
    metrics = Metrics()
    # for i, (inputs, target) in tbar(train_loader):
    for i, (inputs, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.train_gpu:
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        optimizer.zero_grad()

        out3, aux = model(inputs, target)

        main_loss = criterion(out3[:, -1:], target[:, -1:]) + \
                    criterion(out3[:, 0:3:] * out3[:, -1:].data.repeat(1, 3, 1, 1),
                              target[:, 0:3:] * target[:, -1:].data.repeat(1, 3, 1, 1))
        aux_loss = criterion(aux, target[:, -1:]) + 1 - criterion2(out3[:, -1:], target[:, -1:])


        # criterion2
        loss = main_loss + aux_loss*args.aux_weight
        l_tr += loss.item()
        ss += criterion2(out3[:, -1:].data, target[:, -1:].data)

        loss.backward()
        optimizer.step()

        # ----compute metrics----
        n = inputs.size(0)
        metrics.cal_cls_metrics(out3[:, -1:].data, target[:, -1:].data)
        # ----update metrics----
        iou_meter.update(metrics.iou, n)
        main_loss_meter.update(main_loss.item(), n), loss_meter.update(loss.item(), n)
        # ----time----
        batch_time.update(time.time() - end)
        end = time.time()

        # ----learning rate decay----
        current_iter = epoch * num_img_tr + i + 1
        mytrain = AuxTrain(args.base_lr)
        current_lr = mytrain.poly_lr(current_iter, max_iter, power=args.power)
        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        # ----recording results----
        if (i + 1) % args.print_freq == 0:
            info = 'Epoch: [{}/{}][{}/{}]  ' \
                   'Data {data_time.val:.3f} ({data_time.avg:.3f}) '\
                   'Main {loss_meter.val:.4f} '\
                   'IoU {iou_meter.avg:.4f}'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                    data_time=data_time,
                                                    loss_meter=loss_meter, iou_meter=iou_meter)
            print(info, 'ssim:{:.3f}'.format(ss/(i+1)))
        writer.add_scalar('train/loss_iter', loss.item(), i + num_img_tr * epoch)
        writer.add_scalar('train/iou_iter', iou_meter.avg, i + num_img_tr * epoch)

    # ------------vis
    save_image(out3[:, -1:].data.cpu(), saver.vis_dir + '/{}_alpha.png'.format(epoch))
    save_image(target[:, -1:].cpu(), saver.vis_dir + '/{}_gt.png'.format(epoch))
    print('{}/{}'.format(count, num_img_tr))
    return loss_meter.avg, ss/num_img_tr, iou_meter.avg,


def validate(epoch, val_loader, model, criterion, criterion2):
    print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time, data_time = AvgMeter(), AvgMeter()
    loss_meter, iou_meter, auc_meter = AvgMeter(), AvgMeter(), AvgMeter()
    ss = 0.0
    model.eval()
    num_img_tr = len(val_loader)
    end = time.time()
    with torch.no_grad():
        # tbar = tqdm(val_loader, position=0, leave=True)
        # for i, (inputs, target, _) in tbar(val_loader):
        for i, (inputs, target, _) in enumerate(val_loader):
            data_time.update(time.time() - end)
            if args.train_gpu:
                inputs = inputs.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            out = model(inputs)
            out, target = out.data, target.data
            loss = criterion(out[:, -1:], target)
            ss += criterion2(out[:, -1:], target)

            metrics = Metrics()
            metrics.cal_cls_metrics(out[:, -1:], target[:, -1:])
            auc_meter.update(metrics.auc), iou_meter.update(metrics.iou)
            loss_meter.update(loss.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % args.print_freq == 0:
                print('Test: [{}/{}] '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                      'mIoU {iou_meter.avg:.3f}'.format(i + 1, len(val_loader),
                                                        data_time=data_time, batch_time=batch_time,
                                                        loss_meter=loss_meter,
                                                        iou_meter=iou_meter))
    mIoU = iou_meter.avg
    print('Val result: mIoU/ssim {:.4f}/{:.4f}.'.format(mIoU, ss / num_img_tr))
    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    save_image(out[:, -1:].cpu(), saver.vis_dir + '/{}_val.png'.format(epoch))
    # save_image(out[:, 0:3].data.cpu(), saver.vis_dir + '/{}_val_rgb.png'.format(epoch))
    # save_image(aux.cpu(), saver.vis_dir + '/{}_val_aux.png'.format(epoch))
    return loss_meter.avg, mIoU, ss/num_img_tr


if __name__ == '__main__':
    main()
