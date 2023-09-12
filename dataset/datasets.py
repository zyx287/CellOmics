'''
Author: Yuxiang Zhang
Date: 2023-09-08
Description:
    Dataset preparation (load_dataset)
    Dataloader function for different dataset (get_cell_image_dataloader)
Usage:
    1. load_dataset(data_name='emt-cell', num_patch=4, transform_type='offical')
    2. get_cell_image_dataloader(dataset, batch_size=32, k_fold=5, test_shuffle=False, dataset_return=False)
'''
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

def load_dataset(data_name='emt-cell', num_patch=4,
                transform_type='offical'):
    """
    Loads a dataset for training and testing. If augmentloader is used, transform should be None.
    Parameters:
        data_name (str): name of the dataset
        transform_name (torchvision.transform): name of transform to be applied (see data_augu.py)
        train (bool): load training set or not
    Returns:
        dataset (torch.data.dataset)
    """
    _name = data_name.lower()
    _trans_name = transform_type.lower()
    if _name == 'emt-cell':
        from .cellimagedataset import get_cell_image_dataset
        from .data_augu import ContrastiveViewGenerator, ModifiedContrastiveViewGenerator, MaskContrastiveViewGenerator,EMTCombContrastiveViewGenerator
    elif _name == 'ips' or _name == 'ips-comb-featuretest':
        from .cellimagedataset import get_ips_dataset
        from .data_augu import iPSContrastiveViewGenerator, iPSCombContrastiveViewGenerator
    elif _name == 'ips-reg' or _name == 'ips-reg-test':
        from .cellimagedataset import get_ips_reg_dataset
        from .data_augu import iPSContrastiveViewGenerator, iPSCombContrastiveViewGenerator
    else:
        raise NameError(f'No dataset named {data_name}!')
    if _trans_name == 'offical':# EMP-SSL (https://arxiv.org/abs/2304.03977)
        transform = ContrastiveViewGenerator(num_patch=num_patch)
    elif _trans_name == 'custom':
        transform = ModifiedContrastiveViewGenerator(num_patch=num_patch)
    elif _trans_name == 'mask':
        transform = MaskContrastiveViewGenerator(num_patch=num_patch)
    elif _trans_name == 'ips':
        transform = iPSContrastiveViewGenerator(num_patch=num_patch)
    elif _trans_name == 'ips-simple':
        '''
        Define based on random seed 3 (iPS)
        '''
        transform = transforms.Compose([
                        transforms.CenterCrop(150),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.586, 0.588, 0.584], std=[0.038, 0.040, 0.045])
                    ])
    elif _trans_name == 'emt-simple':
        '''
        Define based on random seed 3 (iPS)
        '''
        transform = EMTCombContrastiveViewGenerator(num_patch=num_patch)
    elif _trans_name == 'ips-comb':
        transform = iPSCombContrastiveViewGenerator(num_patch=num_patch)
    else:
        raise NameError(f'No transform named {transform_type}!')

    '''
    transform_strategy: 'normal' or 'mask'
        Controls the transform strategy in cellimagedataset.py
        'normal': 2 outputs (image, label)
        'mask': 3 outputs (image, label, mask_path)
    '''
    if _name == 'emt-cell':
        if _trans_name == 'mask':
            dataset = get_cell_image_dataset(transform=transform, transform_strategy='mask')
        elif _trans_name == 'emt-simple':
            dataset = get_cell_image_dataset(transform=transform, transform_strategy='normal')
        else:
            dataset = get_cell_image_dataset(transform=transform, transform_strategy='normal')
    elif _name == 'ips':
        if _trans_name == 'mask':
            dataset = get_ips_dataset(transform=transform, transform_strategy='mask')
        else:
            dataset = get_ips_dataset(transform=transform, transform_strategy='normal')
    elif _name == 'ips-reg':
        '''
        is_Training = True for (img, coordniates) output
        '''
        if _trans_name == 'mask':
            dataset = get_ips_reg_dataset(transform=transform, transform_strategy='mask')
        else:
            dataset = get_ips_reg_dataset(transform=transform, transform_strategy='normal')
    elif _name == 'ips-reg-test':
        '''
        is_Training = False for (img, coordniates, label) output
        '''
        if _trans_name == 'mask':
            dataset = get_ips_reg_dataset(transform=transform, transform_strategy='mask', is_Training=False)
        else:
            dataset = get_ips_reg_dataset(transform=transform, transform_strategy='normal', is_Training=False)
    elif _name == 'ips-comb-featuretest':
        if _trans_name == 'mask':
            dataset = get_ips_dataset(transform=transform, transform_strategy='mask', is_Training=False)
        else:
            dataset = get_ips_dataset(transform=transform, transform_strategy='normal') 
    else:
        raise NameError('{} Dataset not found'.format(data_name))
    return dataset

def get_cell_image_dataloader(dataset,
                              batch_size=32,
                              k_fold=5,
                              test_shuffle=False,
                              dataset_return=False
                            ):
    if k_fold == 0:
        raise ValueError('k_fold must be greater than 0')
    ratio = 1.0 - 1.0 / k_fold
    train_val_size = int(ratio * len(dataset))
    train_size = int(ratio * train_val_size)
    val_size = train_val_size - train_size
    test_size = len(dataset) - train_val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)# train set must be shuffled
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)# val set must be shuffled
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)
    if dataset_return:
        return train_dataset, val_dataset, test_dataset, train_dl, val_dl, test_dl
    else:
        return train_dl, val_dl, test_dl

def dataset_test(num_patches, batch_size):
    all_dataset = load_dataset("ips-reg-test", num_patch = num_patches, transform_type='ips-comb')
    train_dl, val_dl, test_dl = get_cell_image_dataloader(all_dataset, batch_size=batch_size, k_fold=5)
    for num, data in enumerate(train_dl):
        if num == 0:
            images, _, labels = data
            plt.figure(figsize=(32, 8))
            for i in range(16):
                for j in range(4):
                    plt.subplot(16,4,j+1+i*4)
                    plt.imshow(torch.div(torch.add(images[i][j].view(3, 224, 224),1),2).permute(1,2,0), cmap=None,vmax=1,vmin=0)
                    # plt.tight_layout()
                    plt.axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig('dataset.png',dpi=1200)
        else:
            break

if __name__ == '__main__':
    #######################
    ## Fixed random seed ##
    #######################
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    ips_dataset_test(16, 64)