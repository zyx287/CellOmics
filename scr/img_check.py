'''
Author: Yuxiang Zhang
Date: 2023-09-03
Description:
    Check data augmentation for different dataset
'''
from dataset.datasets import load_dataset, get_cell_image_dataloader, dataset_test

if __name__ == '__main__':
    # all_dataset = load_dataset("ips", train=True, num_patch = 16, transform_type='ips', batch_size=64)
    all_dataset = load_dataset("emt-cell", train=True, num_patch = 16, transform_type='mask', batch_size=64)
    train_dl, val_dl, test_dl = get_cell_image_dataloader(all_dataset, batch_size=64, k_fold=5)
    dataset_test(train_dl)