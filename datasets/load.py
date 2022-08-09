import numpy as np
from PIL import Image
import glob
import cv2
import os
import yaml

from mindspore.dataset.vision.py_transforms import RandomColorAdjust, ToTensor, ToPIL, Normalize
import sys
sys.path.append('.')
from datasets import *


def get_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype('float32')
    return img


def get_bboxes(gt_path, config):
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    polys = []
    dontcare = []
    for line in lines:
        line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        if "#" in gt[-1]:
            dontcare.append(True)
        else:
            dontcare.append(False)
        if config['general']['is_icdar2015']:
            box = [int(gt[i]) for i in range(8)]
        else:
            box = [int(gt[i]) for i in range(len(gt) - 1)]
        polys.append(box)
    return np.array(polys), np.array(dontcare)


class DataLoader():
    def __init__(self, config, isTrain=True):
        self.config = config
        self.isTrain = isTrain

        self.ra = RandomAugment()
        self.ms = MakeSegDetectionData()
        self.mb = MakeBorderMap()

        if isTrain:
            img_paths = glob.glob(os.path.join(config['train']['train_img_dir'],
                                               '*' + config['train']['train_img_format']))
        else:
            img_paths = glob.glob(os.path.join(config['test']['test_img_dir'],
                                               '*' + config['test']['test_img_format']))

        if self.isTrain:
            img_dir = config['train']['train_gt_dir']
            if (config['general']['is_icdar2015']):
                gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.jpg.txt')
                            for img_path in img_paths]
            else:
                gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.txt' )
                            for img_path in img_paths]
        else:
            img_dir = config['test']['test_gt_dir']
            if (config['general']['is_icdar2015']):
                gt_paths = [os.path.join(img_dir, 'gt_' + img_path.split('/')[-1].split('.')[0] + '.txt')
                            for img_path in img_paths]
            else:
                gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.txt' )
                            for img_path in img_paths]

        self.img_paths = img_paths
        self.gt_paths = gt_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        # Getting
        img = get_img(img_path)
        original_img = img
        polys, dontcare = get_bboxes(gt_path, self.config)

        # Random Augment
        if self.isTrain and self.config['train']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, 640)
            img, polys = self.ra.random_rotate(img, polys, self.config['train']['random_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        else:
            polys = polys.reshape((polys.shape[0], polys.shape[1] // 2, 2))

        # Post Process
        if self.isTrain:
            img, gt, gt_mask = self.ms.process(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        else:
            polys = np.array(polys)
            dontcare = np.array(dontcare, dtype=np.bool8)

        # Show Images
        if self.config['general']['is_show']:
            cv2.imwrite('./images/img.jpg', img)
            cv2.imwrite('./images/gt.jpg', gt[0]*255)
            cv2.imwrite('./images/gt_mask.jpg', gt_mask*255)
            cv2.imwrite('./images/thresh_map.jpg', thresh_map*255)
            cv2.imwrite('./images/thresh_mask.jpg', thresh_mask*255)

        # Random Colorize
        if self.isTrain and self.config['train']['is_transform']:
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(ToPIL()(img))

        # Normalize
        img = ToTensor()(img)
        img = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)


        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        else:
            return original_img, img, polys, dontcare


if __name__ == '__main__':
    stream = open('./config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    data_loader = DataLoader(config, isTrain=False)
    import mindspore.dataset as ds
    train_dataset = ds.GeneratorDataset(data_loader, ['img', 'polys', 'dontcare'])
    # train_dataset = ds.GeneratorDataset(data_loader, ['img', 'gt', 'gt_mask', 'thresh_map', 'thresh_mask'])
    train_dataset = train_dataset.batch(1)
    it = train_dataset.create_dict_iterator()
    test = next(it)
    sam = data_loader[19]
    print(sam[0].shape, len(sam[1]), sam[2])