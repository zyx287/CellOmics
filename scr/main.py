'''
Author: Yuxiang Zhang
Date: 2023-09-08
Description:
    training script for CellOmics
'''
############
## Import ##
############
import os
import datetime
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torchvision.transforms.functional as FF
from torch.utils.tensorboard import SummaryWriter

from model.models import encoder, AutoEncoder, Decoder
from model.loss import TotalCodingRate, Similarity_Loss
from model.optim import LARS, LARSWrapper
from dataset.datasets import load_dataset, get_cell_image_dataloader
from util import chunk_avg, cal_TCR# Helper Function

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
print(f'Training Using {device}')

######################
## Parsing Argument ##
######################
parser = argparse.ArgumentParser(description='Unsupervised Learning')

parser.add_argument('--patch_sim', type=int, default=3,
                    help='coefficient of cosine similarity (default: 3)')
parser.add_argument('--tcr', type=float, default=0.,
                    help='coefficient of tcr (default: 0.)')
parser.add_argument('--num_patches', type=int, default=4,
                    help='number of patches used in Region Augmentation (default: 4)')
parser.add_argument('--mse', type=float, default=2,
                    help='coefficient of mse (default: 2)')
parser.add_argument('--arch', type=str, default="resnet18-cell",
                    help='network architecture (default: resnet18-cell)')
parser.add_argument('--bs', type=int, default=64,
                    help='batch size (default: 64)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')        
parser.add_argument('--eps', type=float, default=0.01,
                    help='eps for TCR (default: 0.01)') 
parser.add_argument('--msg', type=str, default="NONE",
                    help='additional message for description (default: NONE)')     
parser.add_argument('--dir', type=str, default="CellOmics-Training",
                    help='directory name (default: CellOmics-Training)')     
parser.add_argument('--data', type=str, default="ips-comb",
                    help='data (default: cell)')          
parser.add_argument('--epoch', type=int, default=30,
                    help='max number of epochs to finish (default: 30)')  
args = parser.parse_args()
num_patches = args.num_patches
date_info = datetime.datetime.now().strftime('%Y%m%d_%H%M')
# Define directory
dir_name = f"/shared/projects/autoencoder/analysis/AnalysisResult/{args.dir}/{date_info}_{args.data}_patchsim{args.patch_sim}_numpatch{args.num_patches}_bs{args.bs}_lr{args.lr}_{args.msg}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
with open(f'{dir_name}/parameter.log','w') as f:
    f.write(f"{args}")
    f.close()
# Tensorboard
log_dir = f"{dir_name}/tensorlog"
os.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir)

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
else:
    train_dataset = load_dataset(args.data, train=True, num_patch = num_patches)
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=2)
# Model1
net = AutoEncoder(encoder(arch = args.arch), Decoder(latent_dim=512))
net = nn.DataParallel(net)
net.to(device)
# Optimizor
learning_rate = 0.01
# opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,nesterov=True)
opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4,nesterov=True)
opt = LARSWrapper(opt,eta=0.005,clip=True,exclude_bias_n_norm=True)
scaler = GradScaler()
# For scheduler (Not necessary)
#---------------------------
if args.data == "imagenet-100":
    num_converge = (150000//args.bs)*args.epoch
elif args.data == "emt-cell":
    num_converge = (len(train_dl.dataset)//args.bs)
elif args.data == "ips":
    num_converge = (len(train_dl.dataset)//args.bs) * 4
else:
    num_converge = (50000//args.bs)*args.epoch
# scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=num_converge, eta_min=0,last_epoch=-1)
#---------------------------

# Loss
contractive_loss = Similarity_Loss()
criterion = TotalCodingRate(eps=args.eps) # For monitoring TCR

#######################
## Training Function ##
#######################
def saving_model(epoch):
    model_dir = dir_name+"/save_models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(net.state_dict(), model_dir+str(epoch)+".pt")

def train_one_epoch(epoch, autoencoder, dataloader, writer):
    if epoch == 20:
        global learning_rate
        learning_rate = 0.001
        for param_group in opt.param_groups:
            param_group['lr'] = learning_rate
    autoencoder.train()
    total_loss_contract = 0.0
    total_loss_TCR = 0.0
    total_loss_MSE = 0.0
    total_loss = 0.0
    for raw_data, label in dataloader:
        autoencoder.zero_grad()
        opt.zero_grad()
        # Images (Augmented as a list[tensor,tensor,...])
        data = torch.cat(raw_data, dim=0)
        data = data.to(device)
        ori_img = raw_data[0]
        print(type(data))
        print(data.size())
        print(data.dtype)
        projection = autoencoder.encoder(data)
        # Chunk Processing
        avg_proj = chunk_avg(projection, num_patches, normalize=True)
        # Contractive Loss
        list_proj = projection.chunk(num_patches, dim=0)
        loss_contract, _ = contractive_loss(list_proj, avg_proj)
        total_loss_contract += loss_contract.item()*(data.size(0)/num_patches)
        loss_TCR = cal_TCR(projection, criterion, num_patches)
        total_loss_TCR += loss_TCR.item()*(data.size(0)/num_patches)
        images = autoencoder.decoder(avg_proj)
        ori_img = ori_img.to(images.device)
        # print('ori_img data on', ori_img.device)
        # print('img on',images.device)
        loss_MSE = F.mse_loss(images, ori_img)
        total_loss_MSE += loss_MSE.item()*(data.size(0)/num_patches)
        loss = args.patch_sim*loss_contract - args.tcr*loss_TCR + args.mse*loss_MSE# Combined loss (Maximal the TCR and similarity)
        total_loss += loss.item()*(data.size(0)/num_patches)
        loss.backward()
        opt.step()
        # scheduler.step()
    total_loss_contract /= len(dataloader.dataset)
    total_loss_TCR /= len(dataloader.dataset)
    total_loss_MSE /= len(dataloader.dataset)
    total_loss /= len(dataloader.dataset)
    writer.add_scalars('Training', {'loss similarity': total_loss_contract, 'loss TCR': total_loss_TCR,'combined loss': total_loss,'mse loss': total_loss_MSE, 'learning rate': opt.param_groups[0]['lr']}, epoch+1)
    print(f'{epoch+1} DONE\n Training Loss: {total_loss} (Simi:{total_loss_contract}, TCR:{total_loss_TCR}, MSE:{total_loss_MSE})')
    return total_loss

def val_one_epoch(epoch, autoencoder, dataloader, writer):
    autoencoder.train()
    total_loss_contract = 0.0
    total_loss_TCR = 0.0
    total_loss_MSE = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for raw_data, label in dataloader:
            # Images (Augmented as a list[tensor,tensor,...])
            data = torch.cat(raw_data, dim=0)
            data = data.to(device)
            ori_img = raw_data[0]
            projection = autoencoder.encoder(data)
            # Chunk Processing
            avg_proj = chunk_avg(projection, num_patches, normalize=True)
            # Contractive Loss
            list_proj = projection.chunk(num_patches, dim=0)
            loss_contract, _ = contractive_loss(list_proj, avg_proj)
            total_loss_contract += loss_contract.item()*(data.size(0)/num_patches)
            loss_TCR = cal_TCR(projection, criterion, num_patches)
            total_loss_TCR += loss_TCR.item()*(data.size(0)/num_patches)
            images = autoencoder.decoder(avg_proj)
            ori_img = ori_img.to(images.device)
            loss_MSE = F.mse_loss(images, ori_img)
            total_loss_MSE += loss_MSE.item()*(data.size(0)/num_patches)
            loss = args.patch_sim*loss_contract - args.tcr*loss_TCR + args.mse*loss_MSE# Combined loss (Maximal the TCR and similarity)
            total_loss += loss.item()*(data.size(0)/num_patches)
    total_loss_contract /= len(dataloader.dataset)
    total_loss_TCR /= len(dataloader.dataset)
    total_loss_MSE /= len(dataloader.dataset)
    total_loss /= len(dataloader.dataset)
    writer.add_scalars('Validate', {'loss similarity': total_loss_contract, 'loss TCR': total_loss_TCR,'combined loss': total_loss,'mse loss': total_loss_MSE, 'learning rate': opt.param_groups[0]['lr']}, epoch+1)
    print(f'{epoch+1} DONE\n Validate Loss: {total_loss} (Simi:{total_loss_contract}, TCR:{total_loss_TCR}, MSE:{total_loss_MSE})')
    return total_loss

def train(autoencoder, train_dl, val_dl, test_dl , writer, num_patches, dir_name):
    train_best = 9999999
    val_best = 999999
    for epoch in range(args.epoch):
        train_loss =  train_one_epoch(epoch, autoencoder, train_dl, writer)
        val_loss = val_one_epoch(epoch, autoencoder, val_dl, writer)
        if train_loss < train_best and val_loss < val_best:
            saving_model(epoch+1)
            train_best = train_loss
            val_best = val_loss
        if train_loss < train_best:
            train_best = train_loss
        if val_loss < val_best:
            val_best = val_loss
    writer.close()

if __name__ == '__main__':
    train(net.module, train_dl, val_dl, test_dl, writer, num_patches, dir_name)

