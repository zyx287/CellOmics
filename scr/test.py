'''
Author: Yuxiang Zhang
Date: 2023-09-10
Description:
    Formated test script for CellOmics
'''
############
## Import ##
############
import datetime
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.models import encoder, AutoEncoder, Decoder
from dataset.datasets import load_dataset, get_cell_image_dataloader
from eval import FeatureEvaluator

#######################
## Fixed random seed ##
#######################
random.seed(3)
np.random.seed(3)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True

####################
## Setting device ##
####################
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Testing Using {device}')

######################
## Parsing Argument ##
######################
parser = argparse.ArgumentParser(description='Unsupervised Learning')

parser.add_argument('--patch_sim', type=int, default=200,
                    help='coefficient of cosine similarity (default: 200)')
parser.add_argument('--tcr', type=float, default=1.,
                    help='coefficient of tcr (default: 1.)')
parser.add_argument('--num_patches', type=int, default=16,
                    help='number of patches used in EMP-SSL (default: 16)')
parser.add_argument('--mse', type=float, default=0.1,
                    help='coefficient of mse (default: 0.1)')
parser.add_argument('--arch', type=str, default="resnet18-cell",
                    help='network architecture (default: resnet18-cell)')
parser.add_argument('--bs', type=int, default=32,
                    help='batch size (default: 32)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')        
parser.add_argument('--eps', type=float, default=0.2,
                    help='eps for TCR (default: 0.2)') 
parser.add_argument('--msg', type=str, default="NONE",
                    help='additional message for description (default: NONE)')     
parser.add_argument('--dir', type=str, default="EMP-SSL-Training",
                    help='directory name (default: EMP-SSL-Training)')     
parser.add_argument('--data', type=str, default="cell",
                    help='data (default: cell)')          
parser.add_argument('--epoch', type=int, default=100,
                    help='max number of epochs to finish (default: 100)')  
args = parser.parse_args()
num_patches = args.num_patches
date_info = datetime.datetime.now().strftime('%Y%m%d_%H%M')
######################
## Prepare Training ##
######################
# DATA
if args.data == "emt-cell":
    all_dataset = load_dataset("emt-cell", num_patch = num_patches, transform_type='emt-simple')
    train_dl, val_dl, test_dl = get_cell_image_dataloader(all_dataset, batch_size=args.bs, k_fold=5)
elif args.data == "ips":
    all_dataset = load_dataset("ips", train=True, num_patch = num_patches, transform_type='ips', batch_size=args.bs)
    train_dl, val_dl, test_dl = get_cell_image_dataloader(all_dataset, batch_size=args.bs, k_fold=5)
elif args.data == 'ips-comb':
    all_dataset = load_dataset("ips-reg", num_patch = num_patches, transform_type='ips-comb')
    train_dl, val_dl, test_dl = get_cell_image_dataloader(all_dataset, batch_size=args.bs, k_fold=5)
elif args.data == 'ips-comb-featuretest':
    all_dataset = load_dataset("ips-comb-featuretest", num_patch = num_patches, transform_type='ips-comb')
    train_dl, val_dl, test_dl = get_cell_image_dataloader(all_dataset, batch_size=args.bs, k_fold=5)
else:
    train_dataset = load_dataset(args.data, train=True, num_patch = num_patches)
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=2)
# Model    
net = AutoEncoder(encoder(arch = args.arch), Decoder(latent_dim=512))
net = nn.DataParallel(net)
net.to(device)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test(net.module, train_dl, val_dl, test_dl, writer, num_patches, dir_name)
    dir_name = f"/shared/projects/autoencoder/analysis/AnalysisResult/CellOmics/20230907_2135_ips-comb_patchsim3_numpatch4_bs64_lr0.01_ViT4Patch"
    with open(f'{dir_name}/parameterTEST.log','w') as f:
        f.write(f"{args}")
        f.close()
    epoch = 2
    net.load_state_dict(torch.load(f'{dir_name}/save_models/{epoch}.pt'))
    net.module.to(device)
    evaluator = FeatureEvaluator(net.module.encoder, train_dl, num_patches, dir_name, epoch-1)
    evaluator()
    epoch = 20
    net.load_state_dict(torch.load(f'{dir_name}/save_models/{epoch}.pt'))
    net.module.to(device)
    evaluator = FeatureEvaluator(net.module.encoder, train_dl, num_patches, dir_name, epoch-1)
    evaluator()