from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

import pdb

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train','val', 'test', 'train_small']
    items = []

    if mode == 'train':
        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)

    elif mode == 'train_small':
        train_img_path = os.path.join(root, 'train_small', 'Img')
        train_mask_path = os.path.join(root, 'train_small', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)

    elif mode == 'val':
        val_img_path = os.path.join(root, 'val', 'Img')
        val_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(val_img_path)
        labels = os.listdir(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(val_img_path, it_im), os.path.join(val_mask_path, it_gt))
            items.append(item)

    else:
        test_img_path = os.path.join(root, 'test', 'Img')
        test_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(test_img_path)
        labels = os.listdir(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(test_img_path, it_im), os.path.join(test_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, augment=False, equalize=False, load_on_gpu=False, load_all_dataset=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.augmentation = augment
        self.equalize = equalize
        self.load_on_gpu = load_on_gpu and torch.cuda.is_available()
        self.load_all_dataset = load_all_dataset and self.load_on_gpu
        self.img_paths = make_dataset(root_dir, mode)
        self.mode = mode

        if self.load_all_dataset:
            self.loaded_imgs = []
            for index in range(self.__len__()):
                img_path, mask_path = self.img_paths[index]
                img = transforms.ToTensor()(Image.open(img_path)).cuda()
                mask = transforms.ToTensor()(Image.open(mask_path).convert('L')).cuda()
                self.loaded_imgs.append((img, mask))

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def augment(img, mask):
        prob = 0.2
        img_size = img[0].size()

        if random() < prob:     # Flip
            img = transforms.functional.vflip(img)
            mask = transforms.functional.vflip(mask)

        if random() < prob:     # Mirror
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        if random() < prob:     # Rotate
            angle = random() * 60 - 30
            img = transforms.functional.rotate(img, angle=angle)
            mask = transforms.functional.rotate(mask, angle=angle)

        if random() < prob:     # Crop
            crop_size = tuple(int((random() + 1) * x / 2) for x in img_size)
            params = transforms.RandomCrop.get_params(img, output_size=crop_size)
            img = transforms.functional.crop(img, *params)
            img = transforms.functional.resize(img, size=img_size)
            mask = transforms.functional.crop(mask, *params)
            mask = transforms.functional.resize(mask, size=img_size)

        if random() < prob:     # Pad
            pad_size = tuple(int(random() * x) for x in img_size)
            img = transforms.Pad(padding=pad_size)(img)
            img = transforms.Resize(size=img_size)(img)
            mask = transforms.Pad(padding=pad_size)(mask)
            mask = transforms.Resize(size=img_size)(mask)

        if random() < prob:     # Z Axe Shift
            scale = random() * 0.25
            params = transforms.RandomPerspective.get_params(*img_size, distortion_scale=scale)
            img = transforms.functional.perspective(img, *params)
            mask = transforms.functional.perspective(mask, *params)

        if random() < prob:     # Brightness shift
            img = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)(img)

        if random() < prob:     # Gaussian Blur
            img = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(img)
        
        if random() < prob:     # Autocontrast
            img = transforms.functional.autocontrast(img)

        if random() < prob:     # Sharpness
            img = transforms.functional.adjust_sharpness(img, sharpness_factor=random()*2)

        return img, mask

    def __getitem__(self, index):
        if self.load_all_dataset:
            img, mask = self.loaded_imgs[index]
            img_path,_ = self.img_paths[index]
        else:
            img_path, mask_path = self.img_paths[index]
            img = transforms.ToTensor()(Image.open(img_path))
            mask = transforms.ToTensor()(Image.open(mask_path).convert('L'))
            if self.load_on_gpu:
                img = img.cuda()
                mask = mask.cuda()

        if self.equalize:
            img = transforms.functional.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        return [img, mask, img_path]
