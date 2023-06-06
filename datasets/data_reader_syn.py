import os, random
# import cv2
import PIL.Image as Image
# import torchvision.transforms as transforms
from util import transform  # pairwise tranforms
from torch.utils.data import Dataset

'''
synthesize while reading
'''

path_root = "../torchCode/data_smk_density/datalist/"
path_train = "blendall.txt"
path_train_gt = "gt_blendall.txt"
path_test = ["SD0{}.txt".format(i + 1) for i in range(3)]
path_test_gt = ["gt_" + item for item in path_test]
# ---- default ----
transform_basic = transform.Compose([
    transform.Resize((256, 256)),
    transform.ToTensor(),
    # transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
data_file = [path_root + path_train, path_root + path_train_gt]  # default


# def is_image_file(filename):
#     filename_lower = filename.lower()
#     return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def load_data(split='train', data_root=None, data_list=None):
    assert split in ['train', 'vis', 'val']
    image_label_list = list()
    smk_list = None
    # --- when data_list = ['data name', 'gt'], which means data/GT name are in separated txt files
    if not (os.path.isfile(data_list[0])):
        raise (RuntimeError("img or GT does not exist: " + data_list + "\n"))
    if len(data_list) == 2:
        if not (os.path.isfile(data_list[0]) and os.path.isfile(data_list[1])):
            raise (RuntimeError("img or GT does not exist: " + data_list + "\n"))
        img_read, label_read = open(data_list[0]).readlines(), open(data_list[1]).readlines()
        smk_list = list()
        if split == 'train':
            for bg in img_read:
                bg = bg.strip()
                bg_name = os.path.join(data_root, bg)
                item = (bg_name, None)
                image_label_list.append(item)
            for smk in label_read:
                smk = smk.strip()
                smk_name = os.path.join(data_root, smk)  # just set place holder for label_name, not for use
                # item = (image_name, label_name, None)
                smk_list.append(smk_name)
        else:
            assert len(img_read) == len(label_read)
            for image, label in zip(img_read, label_read):
                image, label = image.strip(), label.strip()
                image_name = os.path.join(data_root, image)
                label_name = os.path.join(data_root, label)
                item = (image_name, label_name)
                image_label_list.append(item)
    elif len(data_list) == 1:
        if not (os.path.isfile(data_list[0])):
            raise (RuntimeError("img or GT does not exist: " + data_list + "\n"))
        img_read = open(data_list[0]).readlines()
        for line in img_read:
            line = line.strip()
            line_split = line.split()
            image_name = os.path.join(data_root, line_split[0])
            item = (image_name, None)
            image_label_list.append(item)
    else:  # when data_list = 'data name', which means data-GT-pairs or without GT
        Exception("data_list should be a list that contains "
                  "either a pairwise name list or an image name list")

    print("Checking image&label pair {} list done! {} images"
          .format(split, len(image_label_list)))
    return image_label_list, smk_list  # [(img list, gt list)...] or [(img list, None)...]


# data_list = data_name or [data_name, gt] or [data_name, gt, smk]
class Smoke(Dataset):
    def __init__(self, split='train', data_root=path_root, data_list=data_file, transform=transform_basic):
        self.split = split
        self.data_list, self.smk = load_data(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, gt_path = self.data_list[index]
        image = Image.open(image_path)
        # gt = None
        if self.split == 'train':
            idx = random.randint(1, len(self.smk)-1)
            # print(idx)
            gt = Image.open(self.smk[idx])
            # r, g, b, a = gt.split()
            image = Image.composite(gt, image, gt.split()[3])
        else:
            gt = Image.open(gt_path) if gt_path else None
        image, gt = self.transform(image, gt)
        gt = gt if gt is not None else 0  # CHW, HW
        return image, gt, 0.0


