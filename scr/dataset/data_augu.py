'''
Author: zyx287
Date: 2023-08-03
Description:
    Data augmentation for training
    Using different transforms to augment the data
Update: 2023-08-21
    MaskContrastiveViewGenerator: Using cellpose mask for data augmentation
Update: 2023-08-23
    iPSContrastiveViewGenerator: Using central crop for data augmentation
'''
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

'simple'
class SimpleTransform(object):
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, x):# x is a PIL image
        compose = [transforms.Resize(224),
            transforms.ToTensor()]
        if self.normalize:
            compose.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        aug_transform = transforms.Compose(compose)
        augmented = aug_transform(x)
        return augmented

# 'mask'
class MaskContrastiveViewGenerator(object):
    '''
    CellPose mask-based data augmentation for EMT dataset
    '''
    def __init__(self, num_patch, version='v1', normalize=False ,crop_size=60 ,mask_path='/shared/projects/autoencoder/analysis/2023.08.21.MaskedDataset/2023.8.18-EMT-Mask/mask/'):
        self.num_patch = num_patch
        self.mask_path = mask_path
        self.crop_size = crop_size # Size depends on cell images, for EMT, 60 is recommended
        self.normalize = normalize
        self.version = version# Local or Global for Region Augmentation

    def calculate_mask_box(self, mask):
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) > 0:
            points = np.column_stack((mask_indices[1], mask_indices[0]))
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # Calculate the center of the rotated rectangle
            center_x = int(np.mean(box[:, 0]))
            center_y = int(np.mean(box[:, 1]))
            # Calculate the cropping coordinates (Minimal Boundary)
            crop_half_size = self.crop_size // 2
            min_x = max(0, center_x - crop_half_size)
            min_y = max(0, center_y - crop_half_size)
            max_x = min(mask.shape[1] - 1, center_x + crop_half_size)
            max_y = min(mask.shape[0] - 1, center_y + crop_half_size)
            return [min_x, min_y, max_x, max_y]
        else:
            # Using center crop if no mask boundary points
            return None

    def __call__(self, image, img_name):# img_name for getting mask file
        img_mask_path = self.mask_path + img_name + '_mask.tif'
        mask = cv2.imread(img_mask_path, cv2.IMREAD_GRAYSCALE)
        crop_coordinates = self.calculate_mask_box(mask)
        global_compose = [transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()]
        local_compose = [transforms.Resize(512),
            transforms.RandomResizedCrop(224, scale=(0.66, 0.66)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()]
        if self.normalize:
            global_compose.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            local_compose.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        global_aug_transform = transforms.Compose(global_compose)
        local_aug_transform = transforms.Compose(local_compose)
        # Process the image
        if crop_coordinates is not None:
            min_x, min_y, max_x, max_y = crop_coordinates
            cropped_image = image.crop((min_x, min_y, max_x, max_y))
            if self.version == 'v2':
                # Global + Local
                augmented = [global_aug_transform(cropped_image)]
                for _ in range(self.num_patch - 1):
                    augmented.append(local_aug_transform(image))
            elif self.version == 'v1':
                # Local only
                augmented = [local_aug_transform(image) for _ in range(self.num_patch)]
            else:
                raise NameError(f'No version named {self.version} for mask setting!')
            return augmented
        else:
            # If no mask boundary points, return the original image (no augmentation)
            aug_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(300),
                transforms.RandomResizedCrop(224, scale=(0.88, 0.88)),# Ensuring to capture the cell
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            if self.version == 'v2':
                # Global + Local
                augmented = [global_aug_transform(cropped_image)]
                for _ in range(self.num_patch - 1):
                    augmented.append(local_aug_transform(image))
            elif self.version == 'v1':
                # Local only
                augmented = [local_aug_transform(image) for _ in range(self.num_patch)]
            else:
                raise NameError(f'No version named {self.version} for mask setting!')
            return augmented

# 'region'
class RegionAugmentation(object):
    def __init__(self, num_patch=4, version='v1', normalize=False) -> None:
        self.num_patch = num_patch
        self.normalize = normalize
        self.version = version# Local or Global for Region Augmentation

    def __call__(self, x):# x is a PIL image
        '''
        Return: a tensor group of augmented images
        Parameters in Normalize can be calculated using getDatasetStat function (in cellimagedataset.py, seed=3, k_fold=5)
        '''
        if self.version == 'v3':
            global_compose = [transforms.CenterCrop(60),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()]
            local_compose = [transforms.CenterCrop(60),
                                transforms.RandomResizedCrop(224, scale=(0.5, 0.5)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                # transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()]
            if self.normalize:
                global_compose.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                local_compose.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            global_aug_transform = transforms.Compose(global_compose)
            local_aug_transform = transforms.Compose(local_compose)
        else:
            global_compose = [transforms.CenterCrop(80),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()]
            local_compose = [transforms.CenterCrop(100),
                                transforms.RandomResizedCrop(224, scale=(0.8, 0.8)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()]
            if self.normalize:
                global_compose.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
                local_compose.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            global_aug_transform = transforms.Compose(global_compose)
            local_aug_transform = transforms.Compose(local_compose)
        # Process the image
        if self.version == 'v2' or self.version == 'v3':
            # Global + Local
            augmented = [global_aug_transform(x)]
            for _ in range(self.num_patch - 1):
                augmented.append(local_aug_transform(x))
        elif self.version == 'v1':
            # Local only
            augmented = [local_aug_transform(x) for _ in range(self.num_patch)]
        else:
            raise NameError(f'No version named {self.version} for region setting!')
        return augmented


####################################
## Option Augmentation (Not used) ##
####################################
# Modified parameters for EMT dataset using local augmentation
class ModifiedContrastiveViewGenerator(object):
    def __init__(self, num_patch=4) -> None:
        self.num_patch = num_patch

    def __call__(self, x):# x is a PIL image
        '''
        Return: a tensor group of augmented images
        Modified Crop and Resize strategy for EMT dataset
        '''
        aug_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(300),
            transforms.RandomResizedCrop(224, scale=(0.33, 0.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        augmented = [aug_transform(x) for i in range(self.num_patch)]
        return augmented

# Modified parameters for iPS dataset using central crop
class iPSContrastiveViewGenerator(object):
    def __init__(self, num_patch=4) -> None:
        self.num_patch = num_patch

    def __call__(self, x):# x is a PIL image
        '''
        Return: a tensor group of augmented images
        Parameters in Normalize are calculated using getDatasetStat function (in cellimagedataset.py, seed=3, k_fold=5)
        '''
        aug_transform = transforms.Compose([
            transforms.CenterCrop(150),
            transforms.RandomResizedCrop(224, scale=(0.66, 0.66)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.586, 0.588, 0.584], std=[0.038, 0.040, 0.045]),
        ])
        augmented = [aug_transform(x) for i in range(self.num_patch)]
        return augmented

# Modified parameters for iPS dataset using local + global crop
class EMTCombContrastiveViewGenerator(object):
    def __init__(self, num_patch=4) -> None:
        self.num_patch = num_patch

    def __call__(self, x):# x is a PIL image
        '''
        Return: a tensor group of augmented images
        Parameters in Normalize are calculated using getDatasetStat function (in cellimagedataset.py, seed=3, k_fold=5)
        '''
        simple_transform = transforms.Compose([
            transforms.CenterCrop(112),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.586, 0.588, 0.584], std=[0.038, 0.040, 0.045]),
        ])
        aug_transform = transforms.Compose([
            transforms.CenterCrop(80),
            transforms.RandomResizedCrop(224, scale=(0.66, 0.66)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.586, 0.588, 0.584], std=[0.038, 0.040, 0.045]),
        ])
        augmented = [simple_transform(x)]
        for i in range(self.num_patch - 1):
            augmented.append(aug_transform(x))
        return augmented