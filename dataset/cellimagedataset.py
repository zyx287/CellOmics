'''
Author: Yuxiang Zhang
Date: 2023-09-12
Description:
    Dataset preparation for EMT and iPS dataset
    - CellImageDataset: loading for EMT dataset [get_cell_image_dataset]
    - iPSDataset: loading for iPS-RPE dataset [get_ips_dataset]
    - iPSRegDataset: loading for iPS-RPE dataset (with omics data) [get_ips_reg_dataset]
'''
import os
import numpy as np
import random
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

# For EMT Dataset
class CellImageDataset(Dataset):
    def __init__(self, data_root, csv_file, transform=None, transform_strategy='mask'):
        '''
        transform_strategy: 'normal' or 'mask'
          - 'normal': return image and label
          - 'mask': Modified cellpose mask-based transform
        '''
        self.data_root = data_root
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform
        self.transform_strategy = transform_strategy

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = self.data_df.iloc[idx, 0]
        image = Image.open(os.path.join(self.data_root, 'imgs', img_name)).convert('RGB') # Cell images of brightfield microscopy are color images (RGB)
        label = self.data_df.iloc[idx, 1]
        if self.transform_strategy == 'normal':
            if self.transform:
                image = self.transform(image)
        elif self.transform_strategy == 'mask':
            if self.transform:
                image = self.transform(image, img_name)# img_name is the file name for reading corresponding mask
        else:
            raise NameError(f"{self.transform_strategy} is not a valid transform strategy")
        return image, label
    
def get_cell_image_dataset(data_root=r'/shared/projects/autoencoder/rawdata/2023-05-26.EMT.dataset-39.EMT.7-days.remake',
                   csv_file='labels.csv',
                   transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]),
                    transform_strategy='mask'
                ):
    '''
    Loading function for EMT dataset
    transform will be determined in main.py
    '''
    csv_file = os.path.join(data_root, csv_file)
    all_dataset = CellImageDataset(data_root, csv_file, transform=transform, transform_strategy=transform_strategy)
    return all_dataset

# For iPS-RPE Dataset (No Omics Data)
class iPSDataset(Dataset):
    def __init__(self, data_root, csv_file, transform=None, transform_strategy='normal', is_training=True, label_type='label_time'):
        '''
        transform_strategy: 'normal' or 'mask'
          - 'normal': return image and label
          - 'mask': Modified cellpose mask-based transform(Unnecessary for iPS-RPE dataset)
        '''
        self.data_root = data_root
        self.transform = transform
        self.transform_strategy = transform_strategy
        all_dataset = pd.read_csv(csv_file)
        self.data_df = all_dataset
        self.training = is_training # Determine whether to return sample_id (Main key for dataframe)
        self.label_type = label_type # label_time or cell_type or cell_id

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = self.data_df.iloc[idx, 0]
        z_index = self.data_df.iloc[idx, 2]# Mannually selected z_index(Microscope z-stack index, focus planes)
        image = Image.open(os.path.join(self.data_root, f'{z_index}', img_name)).convert('RGB')
        if self.label_type == 'label_time':
            label = self.data_df.iloc[idx, 6]  # label_time
        elif self.label_type == 'cell_type':
            label = self.data_df.iloc[idx, 8]  # cell type
        elif self.label_type == 'cell_id':
            label = self.data_df.iloc[idx, 9]  # cell id
        else:
            raise NameError(f"{self.label_type} is not a valid label type")

        sample_id = self.data_df.iloc[idx, 3] # Main key
        if self.transform_strategy == 'normal':
            if self.transform:
                image = self.transform(image)
        elif self.transform_strategy == 'mask':
            if self.transform:
                image = self.transform(image, img_name)
        else:
            raise NameError(f"{self.transform_strategy} is not a valid transform strategy")
        # Output format
        if self.training:
            return image, label
        else:
            return image, label, sample_id

def get_ips_dataset(data_root=r'/shared/projects/autoencoder/rawdata/2023-08-21.iPSC-RPE/img',
                   csv_file='label_new_scaled.csv',
                   transform = transforms.Compose([
                        transforms.Resize((120, 120)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]),
                    transform_strategy='normal'
                ):
    '''
    Loading function for iPS dataset
    transform will be determined in main.py
    '''
    csv_file = os.path.join(data_root, csv_file)
    all_dataset = iPSDataset(data_root, csv_file, transform=transform, transform_strategy=transform_strategy, is_training = True)
    return all_dataset

# For iPS-RPE Dataset (With Omics Data)
class iPSRegDataset(Dataset):
    def __init__(self, data_root, csv_file, transform=None, transform_strategy='normal', is_training=True, DR = 'PCA', label_type='label_time'):
        self.data_root = data_root
        self.transform = transform
        self.transform_strategy = transform_strategy
        all_dataset = pd.read_csv(csv_file)
        self.data_df = all_dataset# New label
        self.training = is_training
        self.label_type = label_type
        
        # Load Omics data from label.csv
        if DR == 'TSNE':
            self.label_dict = dict(zip(self.data_df['filename'], zip(self.data_df['TSNE1'], self.data_df['TSNE2'])))
        elif DR == 'PCA':
            self.label_dict = dict(zip(self.data_df['filename'], zip(self.data_df['pca1_normalized'], self.data_df['pca2_normalized'])))
        else:
            raise NameError(f"{DR} is not a valid dimentional reduction method")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = self.data_df.iloc[idx, 0]
        z_index = self.data_df.iloc[idx, 2]
        image = Image.open(os.path.join(self.data_root, f'{z_index}', img_name)).convert('RGB')
        file_name = self.data_df.iloc[idx, 0]  # Main key
        # Get corresponding (PC1, PC2) from label_dict (When DR=TSNE, it Dimention 1 and Dimention 2)
        pc1, pc2 = self.label_dict[file_name]
        if self.label_type == 'label_time':
            label = self.data_df.iloc[idx, 6]  # label_time
        elif self.label_type == 'cell_type':
            label = self.data_df.iloc[idx, 8]  # cell type
        elif self.label_type == 'cell_id':
            label = self.data_df.iloc[idx, 9]  # cell id

        if self.transform_strategy == 'normal':
            if self.transform:
                image = self.transform(image)
        elif self.transform_strategy == 'mask':
            if self.transform:
                image = self.transform(image, img_name)
        else:
            raise NameError(f"{self.transform_strategy} is not a valid transform strategy")
        
        if self.training:
            return image, torch.tensor([pc1, pc2], dtype=torch.float32)  # Returning coordinates (PC1, PC2) as a tensor
        else:
            return image, torch.tensor([pc1, pc2], dtype=torch.float32), label # Label for visualization


def get_ips_reg_dataset(data_root=r'/shared/projects/autoencoder/rawdata/2023-08-21.iPSC-RPE/img',
                   csv_file='label_new_scaled.csv',
                   transform = transforms.Compose([
                        transforms.CenterCrop(150),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]),
                    transform_strategy='normal',
                    is_Training = True
                ):
    csv_file = os.path.join(data_root, csv_file)
    all_dataset = iPSRegDataset(data_root, csv_file, transform=transform, transform_strategy=transform_strategy, is_training = is_Training)
    return all_dataset

def getDatasetStat(dataloader):
    print(len(dataloader), len(dataloader.dataset))
    mean = torch.zeros(3)# 3 Channels
    std = torch.zeros(3)
    for data, _ in dataloader:
        for d in range(3):
            mean[d] += data[:,d,:,:].mean()
            std[d] += data[:, d, :, :].std()
    mean.div_(len(dataloader))
    std.div_(len(dataloader))
    return list(mean.numpy()), list(std.numpy())

if __name__ == "__main__":
    #######################
    ## Fixed random seed ##
    #######################
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    mean_std_transforms = transforms.Compose([
            transforms.CenterCrop(150),
            transforms.RandomResizedCrop(224, scale=(0.66, 0.66)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])
    from datasets import get_cell_image_dataloader
    all_dataset = get_ips_dataset(transform=mean_std_transforms, transform_strategy='normal')
    train_dl, val_dl, test_dl = get_cell_image_dataloader(all_dataset, batch_size=1, k_fold=5)
    a, b = getDatasetStat(train_dl)
    c, d = getDatasetStat(val_dl)
    e, f = getDatasetStat(test_dl)
    print('Train',a,b)
    print('Val',c,d)
    print('Test',e,f)
    print('Seed is 3')

