import os, time, logging, sys
sys.path.append("../")
# sys.path.append("../torchCode/MyNet/")
import argparse, cv2
import torch, torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as udata
from torchvision.utils import save_image
from datasets import data_reader_syn as data_reader_me
from util import config
from util import transform as trans
from util.util_smoke import AvgMeter, Metrics, check_dir
from models.mynet_0205_ori import MyNet
from torch.nn.functional import interpolate as it
from util.cal_ssim import SSIM


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
    test: for test set with GT, and testing result will be saved in train_epoch_n.pth/xxx.txt
    test_img: for all kinds of set, and the result images will be saved in SD01/02/03 file
    :return:
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
    # value_scale = 255
    # mean, std = [item * value_scale for item in mean], [item * value_scale for item in std]
    test_trans = trans.Compose([trans.Resize(args.img_size),
                                trans.ToTensor(),
                                trans.Normalize(mean=mean, std=std),
                                ])
    test_data = data_reader_me.Smoke(split='vis', data_root=args.data_root,
                                     data_list=args.test_list, transform=test_trans)
    test_loader = udata.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = MyNet(nclass=args.classes, backbone=args.backbone, pretrained=False,
                  bn=torch.nn.BatchNorm2d)
    device = 'cpu' if not args.train_gpu else 'cuda:{}'.format(args.train_gpu)
    model = torch.nn.DataParallel(model)
    if args.train_gpu:
        model = model.cuda()
    cudnn.benchmark = True
    try:
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=torch.device(device))
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
    iou_meter = AvgMeter()
    # mse_meter = AvgMeter()
    data_time = AvgMeter()
    mse_meter, l = 0.0, torch.nn.MSELoss()
    ss, ssim = 0.0, SSIM(window_size=9)

    # ------- preparing for saving results ----------
    logger.info('--------->>> Start Evaluation ------->>>')
    model.eval()
    data_list = test_data.data_list
    dataset_name = args.test_list[0].split('/')[-1][0:4]
    # -----generating results file path
    base_path = args.model_path.split('.')[0]
    save_path = os.path.join(base_path, dataset_name).replace('\\', '/')
    check_dir(save_path)
    # filename = args.model_path.split('/')[-1].split('.')[0] + '.txt'
    filename = save_path + '.txt'
    # -----computing and writing metrics into text file
    f = open(filename, "w+")
    end = time.time()
    with torch.no_grad():
        for i, (image, target, _) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device)
            data_time.update(time.time() - end)
            # out, _ = model(image)
            out = model(image)
            end = time.time()
        # ------- calculating mse, iou, and ssim ----------
            metrics = Metrics(out[:, -1:].data, target.data, th=0.2)
            metrics.cal_cls_metrics(), iou_meter.update(metrics.iou)
            mse_i = l(out[:, -1:].data, target.data)
            mse_meter += mse_i
            ss_i = ssim(out[:, -1:].data, target.data)
            ss += ss_i
            # ------- writing every result into txt ----------
            image_path, _ = data_list[i]
            image_name = image_path.split('/')[-1].split('.')[0]
            # ----- writing every visualization into image files ---
            if saveImage:
                save_image(out[:, -1],
                           os.path.join(save_path, image_name+'_vis.png').replace('\\', '/'),
                           normalize=False)
            out_single = 'result: name{} ---> mIoU={:.4f}%, MSE={:.4f}, mAcc={:.4f}%, ' \
                         'time={:.4f}s'.format(image_name, iou_meter.val * 100, mse_i, ss_i, data_time.val)
            f.write('\n')
            f.write(out_single)
        # ------- printing results on console ----------
            if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}).'
                            .format(i + 1, len(test_loader), data_time=data_time))

    # ------- mean results of one test set ----------
    logger.info('-------------- End Evaluation ---------------')
    # mIoU, mMse = iouall/len(data_list), mseall/len(data_list)
    mIoU, mTime = iou_meter.avg, data_time.avg
    out_info = 'result: ------> mIoU={:.2f}%, mMse={:.4f}, mSSIM={:.3f}, ' \
               'time={:.4f}s'.format(mIoU*100, mse_meter/1000, ss/1000, mTime)
    logger.info(out_info)

    f.write('\n'), f.write(out_info), f.write('\n')
    f.close()


def test_img(test_loader, test_data, model):
    up_kwargs = {'mode': 'bilinear', 'align_corners': True}

    logger.info('---------- Start Prediction ----------')
    data_time = AvgMeter()
    # ------- preparing for saving results ----------
    model.eval()
    data_list = test_data.data_list
    dataset_name = args.test_list[0].split('/')[-1][0:-4]
    # -----generating results file path
    base_path = args.model_path.split('.')[0]
    save_path = os.path.join(base_path, dataset_name).replace('\\', '/')
    check_dir(save_path)
    filename = save_path + '.txt'
    f = open(filename, "w+")
    end = time.time()
    with torch.no_grad():
        for i, (image, _, _) in enumerate(test_loader):
            data_time.update(time.time() - end)
            _, _, h, w = image.shape
            image = it(image, ((h//2)*2-4, (w//2)*2-4), **up_kwargs)
            out = model(image.to(device))
            # out = out[:, -1:]
        # ------- generating every file name ----------
            image_path, _ = data_list[i]
            image_name = image_path.split('/')[-1].split('.')[0]
            out_single = 'result: name{} ---> time={:.4f}s'.format(image_name, data_time.val)
            logger.info(out_single)
        # ------- saving every estimated smoke image ----------
            out2 = (out[-1].cpu().numpy().transpose(1, 2, 0))
            # ---- ↓ ---black bg---
            save_image(out[:, -1:],
                       os.path.join(save_path, image_name+'_vis.png').replace('\\', '/'),
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
