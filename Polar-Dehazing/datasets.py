import glob
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np


def to_rgb(image):
    rgb_image = Image.new('RGB', image.size)
    rgb_image.paste(image)
    return rgb_image


# several data augumentation strategies
def cv_random_flip(img1, img2, img3, img4, img5):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)
        img4 = img4.transpose(Image.FLIP_LEFT_RIGHT)
        img5 = img5.transpose(Image.FLIP_LEFT_RIGHT)
    return img1, img2, img3, img4, img5


def randomRotation(img1, img2, img3, img4, img5):
    mode = Image.BICUBIC
    angle = [0, 90, 180, 270]
    random_angle = angle[random.randint(0, 3)]
    img1 = img1.rotate(random_angle, mode)
    img2 = img2.rotate(random_angle, mode)
    img3 = img3.rotate(random_angle, mode)
    img4 = img4.rotate(random_angle, mode)
    img5 = img5.rotate(random_angle, mode)
    return img1, img2, img3, img4, img5


def data_argumentation(haze_0, haze_45, haze_90, haze_135, clear):
    # 概率旋转
    haze_0, haze_45, haze_90, haze_135, clear = randomRotation(haze_0, haze_45, haze_90, haze_135, clear)
    # 概率翻转
    haze_0, haze_45, haze_90, haze_135, clear = cv_random_flip(haze_0, haze_45, haze_90, haze_135, clear)
    return haze_0, haze_45, haze_90, haze_135, clear


class HazeDataset(Dataset):
    def __init__(self, path, crop_size=None, mode='train'):
        super(HazeDataset, self).__init__()
        self.crop_size = crop_size
        self.mode = mode
        self.transforms_haze = transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
            transforms.Normalize((0.485, 0.456, 0.406), (0.485, 0.456, 0.406))
        ])
        self.transforms_clear = transforms.ToTensor()

        self.haze_0 = sorted(glob.glob(os.path.join(path, '%s/hazy/0' % mode) + '/*.*'))
        self.haze_45 = sorted(glob.glob(os.path.join(path, '%s/hazy/45' % mode) + '/*.*'))
        self.haze_90 = sorted(glob.glob(os.path.join(path, '%s/hazy/90' % mode) + '/*.*'))
        self.haze_135 = sorted(glob.glob(os.path.join(path, '%s/hazy/135' % mode) + '/*.*'))
        self.clear = sorted(glob.glob(os.path.join(path, '%s/clear' % mode) + '/*.*'))

    def __getitem__(self, item):
        haze_0 = Image.open(self.haze_0[item % len(self.haze_0)])
        haze_45 = Image.open(self.haze_45[item % len(self.haze_45)])
        haze_90 = Image.open(self.haze_90[item % len(self.haze_90)])
        haze_135 = Image.open(self.haze_135[item % len(self.haze_135)])
        clear = Image.open(self.clear[item % len(self.clear)])
        img_size=haze_0.size

        # to RGB
        haze_0 = to_rgb(haze_0)
        haze_45 = to_rgb(haze_45)
        haze_90 = to_rgb(haze_90)
        haze_135 = to_rgb(haze_135)
        clear = to_rgb(clear)

        # crop
        if self.mode == 'train':
            crop_width, crop_height = self.crop_size
            width, height = haze_0.size
            if width < crop_width or height < crop_height:
                name = self.clear[item % len(self.clear)].split('/')[-1]
                raise Exception('Bad image size: {}'.format(name))
            x, y = random.randrange(0, width - crop_width + 1), random.randrange(0, height - crop_height + 1)
            haze_0 = haze_0.crop((x, y, x + crop_width, y + crop_height))
            haze_45 = haze_45.crop((x, y, x + crop_width, y + crop_height))
            haze_90 = haze_90.crop((x, y, x + crop_width, y + crop_height))
            haze_135 = haze_135.crop((x, y, x + crop_width, y + crop_height))
            clear = clear.crop((x, y, x + crop_width, y + crop_height))

        # data augmentation
        if self.mode == 'train':
            haze_0, haze_45, haze_90, haze_135, clear = data_argumentation(haze_0, haze_45, haze_90, haze_135, clear)

        # transforms

        haze_0 = self.transforms_haze(haze_0)
        haze_45 = self.transforms_haze(haze_45)
        haze_90 = self.transforms_haze(haze_90)
        haze_135 = self.transforms_haze(haze_135)
        clear = self.transforms_clear(clear)
        return {'haze_0': haze_0, 'haze_45': haze_45, 'haze_90': haze_90, 'haze_135': haze_135, 'clear': clear}

    def __len__(self):
        return len(self.clear)