'''
Author: zyx287
Date: 2023-09-27
Description:
    Formated training script for CellOmicsV3
'''
############
## Import ##
############
import os
import datetime
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from model.models import encoder, AutoEncoder, Decoder
from model.loss import Similarity_Loss, mutual_loss, TotalCodingRate
from model.optim import LARSWrapper
from dataset.datasets import load_dataset, get_cell_image_dataloader
from util import chunk_avg, cal_TCR, local_chunk_avg# Helper Function

######################
## Parsing Argument ##
######################
parser = argparse.ArgumentParser(description='Unsupervised Learning')
# Loss Function
parser.add_argument('--mse', type=float, default=10,
                    help='coefficient of mse (default: 10), for restoring loss')
parser.add_argument('--patch_sim', type=int, default=3,
                    help='coefficient of cosine similarity (default: 3), original output range: (-1,0)')
parser.add_argument('--mut', type=float, default=3.,
                    help='coefficient of mutual loss (default: 3.)')
parser.add_argument('--dis', type=float, default=1.,
                    help='coefficient of distillation loss (default: 1.)')
parser.add_argument('--tcr', type=float, default=0.,
                    help='coefficient of tcr (default: 0.), only recommended for monitering feature collpasing (not used for training)')
parser.add_argument('--eps', type=float, default=0.2,
                    help='eps for TCR (default: 0.2), also for monitering feature collpasing (not used for training)') 
# Augmentation
parser.add_argument('--num_patches', type=int, default=4,
                    help='number of patches used in Augmentation (default: 4), 4 patch number and 64 batch size for 2 30000MiB GPU')
parser.add_argument('--kfold', type=int, default=5,
                    help='kfold (default: 5)')
parser.add_argument('--transform_type', type=str, default="simple",
                    help='transform_type parameter for dataset-datasets.py-load_dataset (default: simple)',
                    choices=['simple', 'mask-v1', 'mask-v2', 'region-v1', 'region-v2'])
parser.add_argument('--normalize', type=bool, default=False,
                    help='normalize parameter for dataset-datasets.py-load_dataset (default: False)')
# Dataset and Dataloader
parser.add_argument('--dir', type=str, default="CellOmicsV3",
                    help='directory name (default: CellOmicsV3)') 
parser.add_argument('--bs', type=int, default=64,
                    help='batch size (default: 64), middle batch size')
parser.add_argument('--data', type=str, default="ips-cell",
                    help='dataset name (default: ips-cell)')
# Model
parser.add_argument('--arch', type=str, default="resnet18-cellimage",
                    help='network architecture (default: resnet18-cellimage)',
                    choices=['resnet18-cellimage', 'vit-in21k'])     
# Training setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate')
parser.add_argument('--msg', type=str, default="NONE",
                    help='additional message for description (default: NONE)')     
parser.add_argument('--seed', type=int, default=3,
                    help='random seed (default: 3)')
parser.add_argument('--epoch', type=int, default=30,
                    help='max number of epochs to finish (default: 30)')
parser.add_argument('--downstream_task', type=str, default="None",
                    help='downstream task (default: None)',
                    choices=['None', '3typesclassification', '2typesclassification', '3typesPCAregression', 
                             '2typesPCAregression', '3typesLevelregression', '2typesLevelregression'])
parser.add_argument('--ips_down', type=bool, default=False,
                    help='whether using iPS cell downstream task (default: False)')
args = parser.parse_args()
num_patches = args.num_patches
seed = args.seed

#########################
## Saving Path Setting ##
#########################
date_info = datetime.datetime.now().strftime('%Y%m%d_%H%M')
dir_name = f"/shared/projects/autoencoder/analysis/AnalysisResult/CellOmics/{args.dir}/{args.downstream_task}/{date_info}_{args.data}_numpatch{args.num_patches}_bs{args.bs}_{args.msg}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
with open(f'{dir_name}/parameter.log','w') as f:
    f.write(f"{args}")
    f.close()
# Tensorboard
log_dir = f"{dir_name}/tensorlog"
os.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir)

#######################
## Fixed random seed ##
#######################
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

####################
## Setting device ##
####################
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Training Using {device}')


#########################
## Dataset Preparation ##
#########################
if args.data == "emt-cell" or args.data == "ips-cell":
    all_dataset = load_dataset(data_name=args.data, num_patch = num_patches, transform_type=args.transform_type, normalize=args.normalize, downstream_task=args.downstream_task)
    # Leave-one-out cross-validation (3.2 : 0.8 : 1)
    train_dl, val_dl, test_dl = get_cell_image_dataloader(all_dataset, batch_size=args.bs, k_fold=args.kfold)
else:
    raise NameError(f'No dataset named {args.data}, check dataset-datasets.py for more details!')

#######################
## Model Preparation ##
#######################
if (args.downstream_task == 'None') or ((args.downstream_task in ['3typesclassification','2typesclassification','3typesPCAregression','2typesPCAregression','3typesLevelregression','2typesLevelregression']) and (not args.ips_down)):
    # Model1
    print(1)
    model1 = AutoEncoder(encoder(z_dim=512, hidden_dim=512, arch = args.arch), Decoder(latent_dim=512))
    model1 = nn.DataParallel(model1)
    model1.to(device)
    # Optimizor
    learning_rate = 0.01
    opt = optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,nesterov=True)
    opt = LARSWrapper(opt, eta=0.005, clip=True, exclude_bias_n_norm=True)
    # scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=num_converge, eta_min=0,last_epoch=-1)
    # Loss
    restoring_loss = nn.MSELoss()
    feature_loss = Similarity_Loss()
    mutual_loss = mutual_loss()
    monitor = TotalCodingRate(eps=args.eps)
elif (args.downstream_task in ['3typesclassification','2typesclassification','3typesPCAregression','2typesPCAregression','3typesLevelregression','2typesLevelregression']) and args.ips_down:
    # Model1
    print(2)
    model1 = AutoEncoder(encoder(z_dim=512, hidden_dim=512, arch = args.arch), Decoder(latent_dim=512))
    model1 = nn.DataParallel(model1)
    model1.to(device)
    # Model2
    from model.models import MLPRegressionModel
    if args.downstream_task == '3typesclassification':
        model2 = MLPRegressionModel(input_dim=512, hidden_dim=256, output_dim=3)
        criterion = nn.CrossEntropyLoss()
    elif args.downstream_task == '2typesclassification':
        model2 = MLPRegressionModel(input_dim=512, hidden_dim=256, output_dim=2)
        criterion = nn.CrossEntropyLoss()
    elif args.downstream_task == '3typesPCAregression' or args.downstream_task == '2typesPCAregression':
        model2 = MLPRegressionModel(input_dim=512, hidden_dim=256, output_dim=2)
        criterion = nn.L1Loss()
    elif args.downstream_task == '3typesLevelregression' or args.downstream_task == '2typesLevelregression':
        model2 = MLPRegressionModel(input_dim=512, hidden_dim=256, output_dim=10)
        criterion = nn.L1Loss()
    model2.to(device)
    # Optimizor
    learning_rate = 0.01
    opt = optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4,nesterov=True)
    opt = LARSWrapper(opt,eta=0.005,clip=True,exclude_bias_n_norm=True)
else:
    raise NameError('check setting!')
criterion = TotalCodingRate(eps=args.eps)

#######################
## Training Function ##
#######################
def saving_model(model, model_num, epoch, seed):
    if model_num == 1:
        model_dir = dir_name+"/save_model1s/"+seed+"/"
    elif model_num == 2:
        model_dir = dir_name+"/save_model2s/"+seed+"/"
    else:
        raise NameError(f'No model named {model_num}, check train.py for more details!')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir+str(epoch)+".pt")

def train_omics_one_epoch(epoch, autoencoder, dataloader, writer, downstream_task='3typesclassification'):
    if epoch == 10:
        global learning_rate
        learning_rate = 0.001
        for param_group in opt.param_groups:
            param_group['lr'] = learning_rate
    autoencoder.train()
    total_loss = 0.0
    total_loss_feature = 0.0
    total_loss_restoring = 0.0
    total_loss_mutual = 0.0
    total_loss_TCR = 0.0
    for datapair in dataloader:
        if downstream_task == '3typesclassification' or downstream_task == '2typesclassification' or downstream_task == 'None':
            raw_data, label = datapair
        else:
            raw_data, label, sample_id = datapair
        autoencoder.zero_grad()
        opt.zero_grad()
        # Images (Augmented as a list[tensor,tensor,...])
        '''
        raw_data: list[[Global Tensor,tensor,...]
        '''
        data = torch.cat(raw_data, dim=0)
        data = data.to(device)
        ori_img = raw_data[0]
        # print(type(data))
        # print(data.size())
        # print(data.dtype)
        projection = autoencoder.encoder(data)
        # Chunk Processing
        avg_local_proj = local_chunk_avg(projection, num_patches, normalize=True)
        list_proj = projection.chunk(num_patches, dim=0)
        global_proj = list_proj[0]
        # Feature Loss
        loss_feature, _ = feature_loss(list_proj[:], avg_local_proj)
        total_loss_feature += loss_feature.item()*(data.size(0)/num_patches)
        # Mutual Loss
        # print(global_proj.size())
        # print(avg_local_proj.size())
        loss_mutual = mutual_loss(global_proj, avg_local_proj)
        # print(loss_mutual)
        total_loss_mutual += loss_mutual.item()*(data.size(0)/num_patches)
        # Monitor
        loss_TCR = cal_TCR(projection, criterion, num_patches)
        total_loss_TCR += loss_TCR.item()*(data.size(0)/num_patches)
        # Restoring Loss
        images = autoencoder.decoder(global_proj)
        ori_img = ori_img.to(images.device)
        # print('ori_img data on', ori_img.device)
        # print('img on',images.device)
        loss_MSE = restoring_loss(images, ori_img)
        total_loss_restoring += loss_MSE.item()*(data.size(0)/num_patches)
        # Distillation
        global_feature = autoencoder.encoder(ori_img)
        loss_distillation = F.cosine_similarity(global_feature, global_proj)
        total_loss_distillation = loss_distillation.item()*(data.size(0)/num_patches)
        loss = args.patch_sim*loss_feature - args.tcr*loss_TCR + args.mse*loss_MSE + args.mut*loss_mutual + args.dis*loss_distillation# Combined loss (Maximal the TCR and similarity)
        total_loss += loss.item()*(data.size(0)/num_patches)
        loss.backward()
        opt.step()
        # scheduler.step()        
    total_loss_feature /= len(dataloader.dataset)
    total_loss_TCR /= len(dataloader.dataset)
    total_loss_restoring /= len(dataloader.dataset)
    total_loss_mutual /= len(dataloader.dataset)
    total_loss_distillation /= len(dataloader.dataset)
    total_loss /= len(dataloader.dataset)
    writer.add_scalars('Training', {'loss similarity': total_loss_feature, 'loss TCR': total_loss_TCR,'combined loss': total_loss,'mse loss': total_loss_restoring, 'learning rate': opt.param_groups[0]['lr']}, epoch+1)
    print(f'{epoch+1} DONE\n Training Loss: {total_loss} (Simi:{total_loss_feature}, TCR:{total_loss_TCR}, MSE:{total_loss_restoring}, Mutual:{total_loss_mutual}, Distillation:{total_loss_distillation})')
    return total_loss

def val_omics_one_epoch(epoch, autoencoder, dataloader, writer, downstream_task='3typesclassification'):
    autoencoder.eval()
    total_loss = 0.0
    total_loss_feature = 0.0
    total_loss_restoring = 0.0
    total_loss_mutual = 0.0
    total_loss_TCR = 0.0
    for datapair in dataloader:
        if downstream_task == '3typesclassification' or downstream_task == '2typesclassification' or downstream_task == 'None':
            raw_data, label = datapair
        else:
            raw_data, label, sample_id = datapair
        with torch.no_grad():
            for raw_data, label in dataloader:
                # Images (Augmented as a list[tensor,tensor,...])
                '''
                raw_data: list[[Global Tensor,tensor,...]
                '''
                data = torch.cat(raw_data, dim=0)
                data = data.to(device)
                ori_img = raw_data[0]
                projection = autoencoder.encoder(data)
                # Chunk Processing
                avg_local_proj = local_chunk_avg(projection, num_patches, normalize=True)
                list_proj = projection.chunk(num_patches, dim=0)
                global_proj = list_proj[0]
                # Feature Loss
                loss_feature, _ = feature_loss(list_proj[1:], avg_local_proj)
                total_loss_feature += loss_feature.item()*(data.size(0)/num_patches)
                # Mutual Loss
                loss_mutual = mutual_loss(global_proj, avg_local_proj)
                total_loss_mutual += loss_mutual.item()*(data.size(0)/num_patches)
                # Monitor
                loss_TCR = cal_TCR(projection, criterion, num_patches)
                total_loss_TCR += loss_TCR.item()*(data.size(0)/num_patches)
                # Restoring Loss
                images = autoencoder.decoder(global_proj)
                ori_img = ori_img.to(images.device)
                loss_MSE = restoring_loss(images, ori_img)
                total_loss_restoring += loss_MSE.item()*(data.size(0)/num_patches)
                loss = args.patch_sim*loss_feature - args.tcr*loss_TCR + args.mse*loss_MSE + args.mut*loss_mutual# Combined loss (Maximal the TCR and similarity)
                total_loss += loss.item()*(data.size(0)/num_patches)
    total_loss_feature /= len(dataloader.dataset)
    total_loss_TCR /= len(dataloader.dataset)
    total_loss_restoring /= len(dataloader.dataset)
    total_loss_mutual /= len(dataloader.dataset)
    total_loss /= len(dataloader.dataset)
    writer.add_scalars('Validating', {'loss similarity': total_loss_feature, 'loss TCR': total_loss_TCR,'combined loss': total_loss,'mse loss': total_loss_restoring, 'learning rate': opt.param_groups[0]['lr']}, epoch+1)
    print(f'{epoch+1} DONE\n Validating Loss: {total_loss} (Simi:{total_loss_feature}, TCR:{total_loss_TCR}, MSE:{total_loss_restoring})')
    return total_loss

def test_omics_one_epoch(epoch, autoencoder, dataloader, writer, downstream_task='3typesclassification'):
    autoencoder.eval()
    total_loss = 0.0
    total_loss_feature = 0.0
    total_loss_restoring = 0.0
    total_loss_mutual = 0.0
    total_loss_TCR = 0.0
    for datapair in dataloader:
        if downstream_task == '3typesclassification' or downstream_task == '2typesclassification' or downstream_task == 'None':
            raw_data, label = datapair
        else:
            raw_data, label, sample_id = datapair
        with torch.no_grad():
            for raw_data, label in dataloader:
                # Images (Augmented as a list[tensor,tensor,...])
                '''
                raw_data: list[[Global Tensor,tensor,...]
                '''
                data = torch.cat(raw_data, dim=0)
                data = data.to(device)
                ori_img = raw_data[0]
                projection = autoencoder.encoder(data)
                # Chunk Processing
                avg_local_proj = local_chunk_avg(projection, num_patches, normalize=True)
                list_proj = projection.chunk(num_patches, dim=0)
                global_proj = list_proj[0]
                # Feature Loss
                loss_feature, _ = feature_loss(list_proj[1:], avg_local_proj)
                total_loss_feature += loss_feature.item()*(data.size(0)/num_patches)
                # Mutual Loss
                loss_mutual = mutual_loss(global_proj, avg_local_proj)
                total_loss_mutual += loss_mutual.item()*(data.size(0)/num_patches)
                # Monitor
                loss_TCR = cal_TCR(projection, criterion, num_patches)
                total_loss_TCR += loss_TCR.item()*(data.size(0)/num_patches)
                # Restoring Loss
                images = autoencoder.decoder(global_proj)
                ori_img = ori_img.to(images.device)
                loss_MSE = restoring_loss(images, ori_img)
                total_loss_restoring += loss_MSE.item()*(data.size(0)/num_patches)
                # Total loss
                loss = args.patch_sim*loss_feature - args.tcr*loss_TCR + args.mse*loss_MSE + args.mut*loss_mutual# Combined loss (Maximal the TCR and similarity)
                total_loss += loss.item()*(data.size(0)/num_patches)
    total_loss_feature /= len(dataloader.dataset)
    total_loss_TCR /= len(dataloader.dataset)
    total_loss_restoring /= len(dataloader.dataset)
    total_loss_mutual /= len(dataloader.dataset)
    total_loss /= len(dataloader.dataset)
    writer.add_scalars('Testing', {'loss similarity': total_loss_feature, 'loss TCR': total_loss_TCR,'combined loss': total_loss,'mse loss': total_loss_restoring, 'learning rate': opt.param_groups[0]['lr']}, epoch+1)
    print(f'{epoch+1} DONE\n Testing Loss: {total_loss} (Simi:{total_loss_feature}, TCR:{total_loss_TCR}, MSE:{total_loss_restoring})')
    return total_loss

def train_down_one_epoch(epoch, autoencoder, regressor, dataloader, writer):
    if epoch == 10:
        global learning_rate
        learning_rate = 0.001
        for param_group in opt.param_groups:
            param_group['lr'] = learning_rate
    autoencoder.eval()
    regressor.train()
    total_loss_model2 = 0.0
    for datapair in dataloader:
        if args.downstream_task == '2typesclassification' or args.downstream_task == '3typesclassification':
            raw_data, label = datapair
        elif args.downstream_task in ['2typesPCAregression','3typesPCAregression','2typesLevelregression','3typesLevelregression']:
            raw_data, label, cell_label = datapair
        else:
            raw_data, label, sample_id = datapair
        regressor.zero_grad()
        opt.zero_grad()
        # Images (Augmented as a list[tensor,tensor,...])
        data = torch.cat(raw_data, dim=0)
        data = data.to(device)
        label = label.to(device)
        projection = autoencoder.encoder(data)
        # Chunk Processing
        avg_proj = chunk_avg(projection, num_patches, normalize=True)
        coordinates = regressor(avg_proj)
        loss_model2 = criterion(coordinates, label)
        total_loss_model2 += loss_model2.item()*(data.size(0)/num_patches)
        loss_model2.backward()
        opt.step()
        # scheduler.step()
    total_loss_model2 /= len(dataloader.dataset)
    writer.add_scalars('Training', {'mse loss': total_loss_model2, 'learning rate': opt.param_groups[0]['lr']}, epoch+1)
    print(f'{epoch+1} DONE\n Training Loss: {total_loss_model2}')
    return total_loss_model2

def val_down_one_epoch(epoch, autoencoder, regressor, dataloader, writer):
    if epoch == 10:
        global learning_rate
        learning_rate = 0.001
        for param_group in opt.param_groups:
            param_group['lr'] = learning_rate
    autoencoder.eval()
    regressor.eval()
    total_loss_model2 = 0.0
    with torch.no_grad():
        for datapair in dataloader:
            if args.downstream_task == '2typesclassification' or args.downstream_task == '3typesclassification':
                raw_data, label = datapair
            elif args.downstream_task in ['2typesPCAregression','3typesPCAregression','2typesLevelregression','3typesLevelregression']:
                raw_data, label, cell_label = datapair
            else:
                raw_data, label, sample_id = datapair
            # Images (Augmented as a list[tensor,tensor,...])
            data = torch.cat(raw_data, dim=0)
            data = data.to(device)
            label = label.to(device)
            projection = autoencoder.encoder(data)
            # Chunk Processing
            avg_proj = chunk_avg(projection, num_patches, normalize=True)
            coordinates = regressor(avg_proj)
            loss_model2 = criterion(coordinates, label)
            total_loss_model2 += loss_model2.item()*(data.size(0)/num_patches)
    total_loss_model2 /= len(dataloader.dataset)
    writer.add_scalars('Validating', {'mse loss': total_loss_model2, 'learning rate': opt.param_groups[0]['lr']}, epoch+1)
    print(f'{epoch+1} DONE\n Validating Loss: {total_loss_model2}')
    return total_loss_model2

def test_down_one_epoch(epoch, autoencoder, regressor, dataloader, writer):
    if epoch == 10:
        global learning_rate
        learning_rate = 0.001
        for param_group in opt.param_groups:
            param_group['lr'] = learning_rate
    autoencoder.eval()
    regressor.eval()
    total_loss_model2 = 0.0
    with torch.no_grad():
        for datapair in dataloader:
            if args.downstream_task == '2typesclassification' or args.downstream_task == '3typesclassification':
                raw_data, label = datapair
            elif args.downstream_task in ['2typesPCAregression','3typesPCAregression','2typesLevelregression','3typesLevelregression']:
                raw_data, label, cell_label = datapair
            else:
                raw_data, label, sample_id = datapair
            # Images (Augmented as a list[tensor,tensor,...])
            data = torch.cat(raw_data, dim=0)
            data = data.to(device)
            label = label.to(device)
            projection = autoencoder.encoder(data)
            # Chunk Processing
            avg_proj = chunk_avg(projection, num_patches, normalize=True)
            coordinates = regressor(avg_proj)
            loss_model2 = criterion(coordinates, label)
            total_loss_model2 += loss_model2.item()*(data.size(0)/num_patches)
    total_loss_model2 /= len(dataloader.dataset)
    writer.add_scalars('Testing', {'mse loss': total_loss_model2, 'learning rate': opt.param_groups[0]['lr']}, epoch+1)
    print(f'{epoch+1} DONE\n Testing Loss: {total_loss_model2}')
    return total_loss_model2

def train(train_dl, val_dl, test_dl, writer, seed):
    train_best = 99999999
    val_best = 99999999
    if (args.data == 'ips-cell' and (args.downstream_task == '3typesclassification' or args.downstream_task == '2typesclassification') and (not args.ips_down)) or args.downstream_task == 'None':
        for epoch in range(args.epoch):
            train_loss =  train_omics_one_epoch(epoch, model1.module, train_dl, writer)
            val_loss = val_omics_one_epoch(epoch, model1.module, val_dl, writer)
            test_loss = test_omics_one_epoch(epoch, model1.module, test_dl, writer)
            if train_loss < train_best and val_loss < val_best:
                saving_model(model1, 1, epoch+1, seed)
                train_best = train_loss
                val_best = val_loss
            if train_loss < train_best:
                train_best = train_loss
            if val_loss < val_best:
                val_best = val_loss
    elif args.data == 'ips-cell' and (args.downstream_task == '3typesPCAregression' or args.downstream_task == '2typesPCAregression') and args.ips_down:
        for epoch in range(args.epoch):
            train_loss =  train_down_one_epoch(epoch, model2, train_dl, writer)
            val_loss = val_down_one_epoch(epoch, model2, val_dl, writer)
            test_loss = test_down_one_epoch(epoch, model2, test_dl, writer)
            if train_loss < train_best and val_loss < val_best:
                saving_model(model2, 2, epoch+1, seed)
                train_best = train_loss
                val_best = val_loss
            if train_loss < train_best:
                train_best = train_loss
            if val_loss < val_best:
                val_best = val_loss
    writer.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # train(val_dl, train_dl, test_dl, writer, seed)
    train(val_dl, val_dl, val_dl, writer, seed)