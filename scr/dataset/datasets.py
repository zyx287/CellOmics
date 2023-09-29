'''
Author: zyx287
Date: 2023-08-29
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
                transform_type='offical', normalize=False, downstream_task='None', sample_id=False):
    """
    Loads a dataset for training and testing.
    Parameters:
        data_name (str): name of the dataset
            - 'emt-cell': EMT cell image dataset
            - 'ips-cell': iPS cell image dataset
        num_patch (int): number of patches to be extracted from each image (4 is recommended)
        transform_type (str):
            - 'simple': simple transform (only resize and ToTensor)
            - 'mask-v1': using cellpose based mask (only for EMT cell image dataset)
            - 'mask-v2': using cellpose based mask (global + local, only for EMT cell image dataset)
            - 'region-v1': using Region Augmentation (only local, Region Augmentation V1)
            - 'region-v2': using Region Augmentation (global + local, Region Augmentation V2)
        downstream_task (str):
            - 'None': only for feature extraction
            - '3typesclassification': for 3 cell types classification task
            - '2typesclassification': for 2 cell types classification task
            - '3typesPCAregression': for 3 cell coordinates regression task
            - '2typesPCAregression': for 2 cell coordinates regression task
            - '3typesLevelregression': for 3 gene expression level regression task
            - '2typesLevelregression': for 2 gene expression level regression task
    Returns:
        dataset (torch.data.dataset)
    """
    _data_name = data_name.lower()
    _transform_type = transform_type.lower()
    _downstream_task = downstream_task.lower()
    if _transform_type == 'simple':
        from .data_augu import SimpleTransform
        transform = SimpleTransform(normalize=normalize)
    elif _transform_type == 'mask-v1':
        from .data_augu import MaskContrastiveViewGenerator
        transform = MaskContrastiveViewGenerator(num_patch=num_patch, version='v1', normalize=normalize)
    elif _transform_type == 'mask-v2':
        from .data_augu import MaskContrastiveViewGenerator
        transform = MaskContrastiveViewGenerator(num_patch=num_patch, version='v2', normalize=normalize)
    elif _transform_type == 'region-v1':
        from .data_augu import RegionAugmentation
        transform = RegionAugmentation(num_patch=num_patch, version='v1', normalize=normalize)
    elif _transform_type == 'region-v2':
        from .data_augu import RegionAugmentation
        transform = RegionAugmentation(num_patch=num_patch, version='v2', normalize=normalize)
    elif _transform_type == 'region-v3':
        from .data_augu import RegionAugmentation
        transform = RegionAugmentation(num_patch=num_patch, version='v3', normalize=normalize)
    else:
        raise NameError(f'No transform named {transform_type}, check dataset-datasets.py for more details!')
    if _data_name == 'emt-cell':
        if downstream_task != 'None':
            raise ValueError('Downstream task must be None for EMT cell image dataset!')
        from .cellimagedataset import get_cell_image_dataset
        if _transform_type == 'mask-v1' or _transform_type == 'mask-v2':
            dataset = get_cell_image_dataset(transform=transform, transform_output='mask', downstream_task=downstream_task)
        else:
            dataset = get_cell_image_dataset(transform=transform, transform_output='simple', downstream_task=downstream_task)
    elif _data_name == 'ips-cell':
        if downstream_task == 'None':
            raise ValueError('Downstream task must be specified for iPS cell image dataset!')
        else:
            from .cellimagedataset import get_ips_dataset
            dataset = get_ips_dataset(transform=transform, downstream_task=downstream_task, sample_id=sample_id)
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

def dataset_test():
    dataloader = load_dataset('cell', train=True, num_patch=16, transform_type='custom',batch_size=4)
    for num, data in enumerate(dataloader):
        if num == 0:
            images, labels = data
            # print(len(images))
            # print(len(labels))
            # print(images[0][0].shape)
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

def ips_dataset_test(num_patches, batch_size):
    all_dataset = load_dataset("ips-reg-test", num_patch = num_patches, transform_type='ips-comb')
    train_dl, val_dl, test_dl = get_cell_image_dataloader(all_dataset, batch_size=batch_size, k_fold=5)
    for num, data in enumerate(train_dl):
        if num == 0:
            images, _, labels = data
            # print(len(images))
            # print(len(labels))
            # print(images[0][0].shape)
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

def dataset_saving_test():
    pass


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