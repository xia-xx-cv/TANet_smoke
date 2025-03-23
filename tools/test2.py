import time, os
import logging
import argparse
import torch, torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as udata
from torchvision.utils import save_image
from datasets import data_reader_syn as data_reader
from util import config
from util import transform as trans
from util.util_smoke import AvgMeter, Metrics, check_dir
from models.mynet_0205_ori import MyNet
from torch.nn.functional import interpolate as it
from util.cal_ssim import SSIM
from tqdm import tqdm


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('run/smoke.txt', encoding='utf-8')
    fmt = "[%(asctime)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(description='Dual smoke density estimation')
    parser.add_argument('--config', type=str, default='smoke_ori.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def main():
    """
    pred_split = 'test_gt'(call test_gt function) or 'test_img'(call test_img function)
    test_gt: for test set with GT, and testing result will be saved in train_epoch_n.pth/xxx.txt
    test_img: for all kinds of set, and the result images will be saved in SD01/02/03 file
    """
    global args, logger, device
    args = get_parser()
    logger = get_logger()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    # logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    # ---- args.dataset == 'smoke'-----------
    mean, std = [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]
    test_trans = trans.Compose([
        trans.Resize(),
        trans.ToTensor(),
        trans.Normalize(mean=mean, std=std)
    ])
    test_data = data_reader.Smoke(split='val', data_root=args.data_root,
                                 data_list=args.test_list, transform=test_trans)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_val,
                                             shuffle=False, num_workers=args.workers,
                                             pin_memory=True, sampler=None)
    device = 'cpu' if not args.train_gpu else 'cuda'

    model = MyNet(nclass=args.classes, backbone=args.backbone, pretrained=False,
                   bn=torch.nn.BatchNorm2d).eval().to(device)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    try:
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    except Exception as err:
        print(err)
        print("=> no checkpoint found at '{}'".format(args.model_path))
        exit(-1)

    # tmp = args.model_path[args.model_path.find('_') + 1: args.model_path.rfind('.pth')]  # epoch num
    save_folder = args.model_path[0:args.model_path.rfind('train_epoch')]
    check_dir(save_folder)
    if len(args.test_list) == 1:
        # names = [line.rstrip('\n') for line in open(args.test_list[0])]
        test_img(test_loader, test_data, model)
    else:
        test_gt(test_loader, test_data, model, saveImage=False)


def test_gt(test_loader, test_data, model, saveImage=False):
    # ------- with GT ----------
    data_time = AvgMeter()
    mse_loss = torch.nn.MSELoss()
    ssim = SSIM(window_size=9)
    mse_meter, iou_meter, ss_meter = AvgMeter(), AvgMeter(), AvgMeter()

    # ------- preparing for saving results ----------
    logger.info('--------->>> Start Evaluation ------->>>')
    data_list = test_data.data_list
    dataset_name = args.test_list[0].split('/')[-1][0:4]
    # -----generating results file path
    base_path = args.model_path.split('.pth')[0]
    save_path = os.path.join(base_path, dataset_name).replace('\\', '/')
    check_dir(save_path)
    # filename = args.model_path.split('/')[-1].split('.')[0] + '.txt'
    filename = save_path + '.txt'
    # -----computing and writing metrics into text file
    f = open(filename, "w+")
    end = time.time()
    metric = Metrics(th=0.2)
    with torch.no_grad():
        for i, (image, target, _) in tqdm(enumerate(test_loader)):
            image = image.to(device)
            target = target.to(device)
            data_time.update(time.time() - end)
            out = model(image)
            end = time.time()
            # ------- calculating mse, iou, and ssim ----------
            metric.cal_cls_metrics(out[:, -1:], target[:, -1:])
            iou_i = metric.iou
            mse_i = mse_loss(out[:, -1:].data, target.data)
            ss_i = ssim(out[:,-1::].data.cpu(), target.data.cpu())
            mse_meter.update(mse_i), iou_meter.update(iou_i), ss_meter.update(ss_i)

            # ------- writing every result into txt ----------
            image_path, _ = data_list[i]
            image_name = image_path.split('/')[-1].split('.')[0]
            out_single = 'result: name{} ---> mIoU={:.4f}%, MSE={:.4f}, mAcc={:.4f}%, ' \
                         'time={:.4f}s'.format(image_name, iou_i * 100, mse_i, ss_i, data_time.val)
            f.write('\n')
            f.write(out_single)
    # ------- mean results of one test set ----------
    logger.info('-------------- End Evaluation ---------------')
    miou, mmse, ss = iou_meter.avg, mse_meter.avg, ss_meter.avg
    out_info = 'result: ------> mIoU={:.2f}%, mMse={:.4f}, mSSIM={:.3f}, ' \
               'time={:.4f}s'.format(miou * 100, mmse, ss, data_time.avg)
    logger.info(out_info)

    f.write('\n'), f.write(out_info), f.write('\n')
    f.close()


def test_img(test_loader, test_data, model):
    up_kwargs = {'mode': 'bilinear', 'align_corners': True}

    logger.info('---------- Start Prediction ----------')
    data_time = AvgMeter()
    # ------- preparing for saving results ----------
    data_list = test_data.data_list
    dataset_name = args.test_list[0].split('/')[-1][0:-4]
    # -----generating results file path
    base_path = args.model_path.split('.pth')[0]
    save_path = os.path.join(base_path, dataset_name).replace('\\', '/')
    check_dir(save_path)
    filename = save_path + '.txt'
    f = open(filename, "w+")
    end = time.time()
    with torch.no_grad():
        for i, (image, _, _) in tqdm(enumerate(test_loader)):
            data_time.update(time.time() - end)
            _, _, h, w = image.shape
            # image = it(image, (int(h//2)*2, int(w//2)*2), **up_kwargs)
            image = it(image, (512, 512), **up_kwargs)
            out = model(image.cuda())
            # ------- generating every file name ----------
            image_path, _ = data_list[i]
            image_name = image_path.split('/')[-1].split('.')[0]
            out_single = 'result: name{} ---> time={:.4f}s'.format(image_name, data_time.val)
            logger.info(out_single)
            out = it(out[:,-1::].data.cpu(), (h, w), **up_kwargs)
            save_image(out,
                       os.path.join(save_path, image_name + '_vis.png').replace('\\', '/'),
                       normalize=False)

    logger.info('---------- End Prediction ------------')
    single_time, all_time = data_time.avg, data_time.sum
    out_info = 'result: ------> single_time={:.4f}s, all_time={:.4f}s'.format(single_time, all_time)
    logger.info(out_info)
    f.write('\n')
    f.write(out_info)
    f.write('\n')
    f.close()

if __name__ == '__main__':
    main()
