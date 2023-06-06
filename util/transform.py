import random
import PIL.Image as Image
import numbers
import math
import collections
from torchvision.transforms import functional as F

__all__ = ["Compose", "ToTensor", "Normalize", "Resize",
           "RandomHorizontalFlip", "RandomVerticalFlip",
           "RandomScale", "RandomRotation", "RandomResizedCrop"]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',  # -- segmenation gt
    Image.BILINEAR: 'PIL.Image.BILINEAR',  # --img and regression gt
}


class Compose(object):
    # Composes trans: trans.Compose([trans.RandScale([0.5, 2.0]), trans.ToTensor()])
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img, gt):
        for t in self.trans:
            img, gt = t(img, gt)
        return img, gt


class ToTensor(object):
    # Converts (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, img, gt=None):
        img = F.to_tensor(img)
        gt = F.to_tensor(gt) if gt is not None else gt
        return img, gt


class Normalize(object):
    """ Normalize synthesized smoke images or backgrounds
        with m and std deviation along channel: ch = (ch - m) / std.
        Smokes and targets remain unchanged.
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean  # [item*255 for item in mean]
        self.std = std  # [item*255 for item in std]

    def __call__(self, img, gt):
        # image.sub_(mean[:, None, None]).div_(std[:, None, None])
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return img, gt


class Resize(object):
    """ Resize images but for targets/smokes only
    """
    def __init__(self, size=256, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, gt=None):
        img = F.resize(img, self.size, self.interpolation)
        gt = F.resize(gt, self.size, self.interpolation) if gt is not None else gt
        # smk = F.resize(smk, self.size, self.interpolation) if smk is not None else smk
        return img, gt


class RandomScale(object):
    """ Resize the smoke images or backgrounds to the given size, size: (h, w)*.
        Only applied to smokes and GTs. """
    def __init__(self, scale_range, p=0.5, interpolation=Image.BILINEAR):
        assert (isinstance(scale_range, collections.Iterable) and len(scale_range) == 2)
        self.scale_range = scale_range
        self.p = p
        self.interpolation = interpolation

    @staticmethod
    def get_params(scale_range):
        # scale = []
        # scale[0] = random.uniform(scale_range[0], scale_range[1])
        # scale[1] = random.uniform(scale_range[0], scale_range[1])
        scale = random.uniform(scale_range[0], scale_range[1])
        return scale

    def __call__(self, img, gt=None):
        if random.random() < self.p:
            scale = self.get_params(self.scale_range)
            # scale = [item*256 for item in scale]
            img = F.resize(img, scale // 1, self.interpolation)
            if gt is not None:
                gt = F.resize(gt, scale//1, self.interpolation)
            # if smk is not None:
            #     smk = F.resize(smk, scale//1, self.interpolation)
        return img, gt


class RandomHorizontalFlip(object):
    """ Randomly flip smoke images/bgs and targets/smks
        under different probabilities to gain diversity.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt=None):
        if random.random() < self.p:
            img = F.hflip(img)
            gt = F.hflip(gt) if gt is not None else gt
        return img, gt


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt=None):
        if random.random() < self.p:
            img = F.vflip(img)
            gt = F.vflip(gt) if gt is not None else gt
        return img, gt


class RandomRotation(object):
    """  Only applied to smoke images/backgrounds """
    def __init__(self, degrees=90, p=0.5, center=None):
        if isinstance(degrees, numbers.Number):
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of length 2.")
            self.degrees = degrees
        self.p = p
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.randint(degrees[0], degrees[1])
        return angle

    def __call__(self, img, gt=None):
        if random.uniform(0, 1) > self.p:
            angle = self.get_params([item//2 for item in self.degrees])
            img = F.rotate(img, angle, False, False, self.center)
            if gt is not None:
                gt = F.rotate(gt, angle, False, False, self.center)
        return img, gt


class RandomResizedCrop(object):
    """ scale first (implemented by resize), then crop to the given size (default 256x256)
    """
    def __init__(self, size=256, scale=(0.6, 1.2), ratio=(3. / 4., 4. / 3.), p=0.5, interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5 and min(ratio) <= (h / w) <= max(ratio):
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            i = random.randint(0, img.size[1] - h)
            j = random.randint(0, img.size[0] - w)
            return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, gt=None):
        if self.p < 0.5:
            # i, j, h, w = self.get_params(img, self.scale, self.ratio)
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            img = F.resize(img, (h, w), self.interpolation)
            img = F.crop(img, i, j, self.size[0], self.size[1])
            if gt is not None:
                gt = F.resize(gt, (h, w), self.interpolation)
                gt = F.crop(gt, i, j, self.size[0], self.size[1])
        return img, gt


class SmkTransform(object):
    def __init__(self, translate=[random.randint(-30, 30), random.randint(-50, 50)],
                 scale=random.uniform(0.6, 1.2), angle=90, shear=None, resample=0, fillcolor=0):
        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "Argument translate should be a list or tuple of length 2"
        self.angle = angle*random.randint(-2, 2)
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.kwargs = {"fillcolor": fillcolor}

    def __call__(self, img=None, gt=None):
        if gt is not None:
            output_size = gt.size
            center = (gt.size[0] * 0.5 + 0.5, gt.size[1] * 0.5 + 0.5)
        if self.translate[0] >= output_size[0]//2:
            self.translate[0] = random.randint(-output_size[0]//3, output_size[0]//3)
        if self.translate[1] >= output_size[1]//2:
            self.translate[1] = random.randint(-output_size[1]//3, output_size[1]//3)

        angle = math.radians(self.angle)
        shear = math.radians(self.shear) if self.shear is not None else 0
        scale = 1.0 / self.scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + \
            math.sin(angle + shear) * math.sin(angle)
        matrix = [math.cos(angle + shear), math.sin(angle + shear), 0,
                  -math.sin(angle), math.cos(angle), 0]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - self.translate[0])\
                     + matrix[1] * (-center[1] - self.translate[1])
        matrix[5] += matrix[3] * (-center[0] - self.translate[0])\
                     + matrix[4] * (-center[1] - self.translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        if gt is not None:
            gt = gt.transform(output_size, Image.AFFINE, matrix, self.resample, **self.kwargs)
        return img, gt


if __name__ == '__main__':
    img = Image.open('E:/torchCode/data_smk_density/blendall2/00000_smk.png')
    tr_trans = Compose([
        SmkTransform(),
        ToTensor(),
        Normalize()])
    img2, _, _ = tr_trans(None, img)
    img.save('c://a.png')
    # print(img2.max(), img2.min(), img2.mean())
    from torchvision.utils import save_image
    save_image(img2.unsqueeze(0), 'c:/a2.png', normalize=False)
