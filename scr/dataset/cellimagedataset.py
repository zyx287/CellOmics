'''
Author: zyx287
Date: 2023-08-04
Description:
    Dataset preparation
    Cell image for different dataset
Update: 2023-08-23
    iPSDataset: loading for iPS-RPE dataset
    getDatasetStat(): calculating the mean and std for specific dataset
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
class EMTCellImageDataset(Dataset):
    def __init__(self, data_root, csv_file, transform=None, transform_output='mask', downstream_task='None'):
        self.data_root = data_root
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform
        self.transform_output = transform_output
        self.downstream_task = downstream_task

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = self.data_df.iloc[idx, 0]
        image = Image.open(os.path.join(self.data_root, 'imgs', img_name)).convert('RGB') # Cell images of brightfield microscopy are color images (RGB)
        # print("Image shape before transformation:", image.size)
        label = self.data_df.iloc[idx, 1]
        if self.transform_output == 'simple':
            if self.transform:
                image = self.transform(image)
        elif self.transform_output == 'mask':
            if self.transform:
                image = self.transform(image, img_name)# img_name is the file name for reading corresponding mask
        else:
            raise NameError(f"{self.transform_strategy} is not a valid transform strategy")
        # print("Image shape after transformation", image.shape)
        if self.downstream_task == 'None':
            return image, label
        else:
            raise NameError(f"{self.downstream_task} is not a valid downstream task for EMT cell")
    
def get_cell_image_dataset(data_root=r'/shared/projects/autoencoder/rawdata/2023-05-26.EMT.dataset-39.EMT.7-days.remake',
                   csv_file='labels.csv',
                   transform = transforms.Compose([
                        transforms.Resize((120, 120)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]),
                    transform_output='mask',
                    downstream_task='None'
                ):
    '''
    Loading function for EMT dataset
    '''
    csv_file = os.path.join(data_root, csv_file)
    all_dataset = EMTCellImageDataset(data_root, csv_file, transform=transform, transform_output=transform_output, downstream_task=downstream_task)
    return all_dataset

# For iPS-RPE Dataset
class iPSImageDataset(Dataset):
    def __init__(self, data_root, csv_file, transform=None, transform_output='simple',downstream_task='3typesclassification', label_type='label_time', sample_id=False):
        self.data_root = data_root
        self.transform = transform
        self.transform_output = transform_output
        all_dataset = pd.read_csv(csv_file)
        self.data_df = all_dataset# New label
        self.sample_id = sample_id # Determine whether to return sample_id (Main key for dataframe)
        self.downstream_task = downstream_task
        self.label_type = label_type
        if self.downstream_task == '2typesclassification' or self.downstream_task == '2typesPCAregression' or self.downstream_task == '2typesLevelregression':
            self.data_df = self.data_df[self.data_df['cell_kind'] == 0]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if self.downstream_task == '3typesclassification' or self.downstream_task == '3typesPCAregression' or self.downstream_task == '3typesLevelregression':
            img_name = self.data_df.iloc[idx, 0]
            z_index = self.data_df.iloc[idx, 2]
            image = Image.open(os.path.join(self.data_root, f'{z_index}', img_name)).convert('RGB')
            sample_id = self.data_df.iloc[idx, 3] # Main key
            if self.transform_output == 'simple':
                if self.transform:
                    image = self.transform(image)
            elif self.transform_strategy == 'mask':
                if self.transform:
                    image = self.transform(image, img_name)
            # print("Image shape before transformation:", image.size)
            if self.label_type == 'label_time':
                label = self.data_df.iloc[idx, 6]  # label_itime
            elif self.label_type == 'cell_id':
                label = self.data_df.iloc[idx, 9]  # cell id
            elif self.label_type == 'cell_kind':
                label = self.data_df.iloc[idx, 8]  # cell kind
            else:
                raise NameError(f"{self.label_type} is not a valid label type for iPS-RPE dataset")
        elif self.downstream_task == '2typesclassification' or self.downstream_task == '2typesPCAregression' or self.downstream_task == '2typesLevelregression':
            img_name = self.data_df.iloc[idx, 0]
            z_index = self.data_df.iloc[idx, 2]
            image = Image.open(os.path.join(self.data_root, f'{z_index}', img_name)).convert('RGB')
            sample_id = self.data_df.iloc[idx, 3] # Main key
            if self.transform_output == 'simple':
                if self.transform:
                    image = self.transform(image)
            elif self.transform_strategy == 'mask':
                if self.transform:
                    image = self.transform(image, img_name)
            if self.label_type == 'label_time':
                label = self.data_df.iloc[idx, 6]
                # For fitting classification loss function
                if label == 2:# Week6
                    label = 0
                elif label == 3:# Week12
                    label = 1
            else:
                raise NameError(f"{self.label_type} is not a valid label type for iPS-RPE dataset 2 cell types downstream task")
        else: 
            raise NameError(f"{self.downstream_task} is not a valid downstream task for iPS-RPE dataset(PS: None could be replaced by 3typesclassification or 2typesclassification)")
        if self.downstream_task == '3typesPCAregression' or self.downstream_task == '2typesPCAregression':
            pc1 = self.data_df.iloc[idx, 4]
            pc2 = self.data_df.iloc[idx, 5]
            if self.sample_id:
                return image, torch.tensor([pc1, pc2], dtype=torch.float32), label, sample_id
            else:
                return image, torch.tensor([pc1, pc2], dtype=torch.float32), label
        elif self.downstream_task == '3typesLevelregression' or self.downstream_task == '2typesLevelregression':
            genes = []
            for i in range(12, 22):
                # print(i)
                genes.append(self.data_df.iloc[idx, i])
            if self.sample_id:
                return image, torch.tensor(genes, dtype=torch.float32), label, sample_id
            else:
                return image, torch.tensor(genes, dtype=torch.float32), label
        elif self.downstream_task == '3typesclassification' or self.downstream_task == '2typesclassification':
            if self.sample_id:
                return image, label, sample_id
            else:
                return image, label
            
def get_ips_dataset(data_root=r'/shared/projects/autoencoder/rawdata/2023-08-21.iPSC-RPE/img',
                   csv_file='label_new_allinone.csv',
                   transform = transforms.Compose([
                        transforms.Resize((120, 120)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]),
                    transform_output='simple',
                    downstream_task='3typesclassification',
                    label_type='label_time',
                    sample_id=False
                ):
    csv_file = os.path.join(data_root, csv_file)
    all_dataset = iPSImageDataset(data_root, csv_file, transform=transform, transform_output=transform_output, downstream_task=downstream_task, label_type=label_type, sample_id=sample_id)
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

