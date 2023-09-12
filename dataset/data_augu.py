'''
Author: Yuxiang Zhang
Date: 2023-09-08
Description:
    Data augmentation for training
    - ContrastiveViewGenerator: Sample crop, flip, grayscale, resize, normalize
    - ModifiedContrastiveViewGenerator: Using central crop for EMT dataset (fine-tuned parameters)
    - iPSContrastiveViewGenerator: Using central crop and mean-std for iPS dataset (fine-tuned parameters)
    - MaskContrastiveViewGenerator: Using cellpose mask for EMT data augmentation (Not necessary for iPS dataset)
    - iPSCombContrastiveViewGenerator: Using central crop for i+n augmentation for iPS dataset [considering global feature] (fine-tuned parameters)
    - EMTCombContrastiveViewGenerator: Using central crop for i+n augmentation for EMT dataset [considering global feature] (fine-tuned parameters)
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

class ContrastiveViewGenerator(object):
    '''
    Mostly copy-paste from https://arxiv.org/abs/2304.03977
    '''
    def __init__(self, num_patch=4) -> None:
        self.num_patch = num_patch

    def __call__(self, x):# x is a PIL image
        '''
        Return: a tensor group of augmented images
        Sample strategy for all dataset
        '''
        aug_transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.25, 0.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        augmented = [aug_transform(x) for i in range(self.num_patch)]
        return augmented

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
    
class iPSContrastiveViewGenerator(object):
    def __init__(self, num_patch=4) -> None:
        self.num_patch = num_patch

    def __call__(self, x):# x is a PIL image
        '''
        Return: a tensor group of augmented images
        Parameters in Normalize are calculated using getDatasetStat function (in cellimagedataset.py, seed=3, k_fold=5)
        '''
        aug_transform = transforms.Compose([
            transforms.CenterCrop(90),
            transforms.RandomResizedCrop(224, scale=(0.66, 0.66)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.586, 0.588, 0.584], std=[0.038, 0.040, 0.045]),
        ])
        augmented = [aug_transform(x) for i in range(self.num_patch)]
        return augmented

class MaskContrastiveViewGenerator(object):
    '''
    CellPose mask-based data augmentation for EMT dataset
    '''
    def __init__(self, num_patch, mask_path='/shared/projects/autoencoder/analysis/2023.08.21.MaskedDataset/2023.8.18-EMT-Mask/mask/') -> None:
        self.num_patch = num_patch
        self.mask_path = mask_path
        self.crop_size = 60

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
            return None

    def __call__(self, image, img_name):  # Pass the image and mask_path
        img_mask_path = self.mask_path + img_name + '_mask.tif'
        mask = cv2.imread(img_mask_path, cv2.IMREAD_GRAYSCALE)
        crop_coordinates = self.calculate_mask_box(mask)
        
        if crop_coordinates is not None:
            min_x, min_y, max_x, max_y = crop_coordinates
            cropped_image = image.crop((min_x, min_y, max_x, max_y))
            aug_transform = transforms.Compose([
                transforms.Resize(512),
                # transforms.CenterCrop(300),
                transforms.RandomResizedCrop(224, scale=(0.33, 0.33)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            augmented = [aug_transform(cropped_image) for _ in range(self.num_patch)]
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
            augmented = [aug_transform(image) for _ in range(self.num_patch)]
            return augmented

class iPSCombContrastiveViewGenerator(object):
    def __init__(self, num_patch=4) -> None:
        self.num_patch = num_patch

    def __call__(self, x):# x is a PIL image
        '''
        Return: a tensor group of augmented images
        Parameters in Normalize are calculated using getDatasetStat function (in cellimagedataset.py, seed=3, k_fold=5)
        '''
        simple_transform = transforms.Compose([
            transforms.CenterCrop(90),
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
