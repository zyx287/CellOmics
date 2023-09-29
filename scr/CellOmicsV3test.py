'''
Author: zyx287
Date: 2023-08-21
Description:
    Formated training script for EMP-SSL on cell image
'''
############
## Import ##
############
import os
import re
import random
import argparse
import numpy as np

import torch
import torch.nn as nn

from model.models import encoder, AutoEncoder, Decoder
from dataset.datasets import load_dataset, get_cell_image_dataloader
from util import chunk_avg, cal_TCR, local_chunk_avg# Helper Function
from model.eval import FeatureEvaluator, RegressionEvaluator

#########################
## Saving Path Setting ##
#########################
# Manual Setting
dir_name = f"/shared/projects/autoencoder/analysis/AnalysisResult/CellOmics/CellOmicsV3/2typesclassification/20230929_1651_ips-cell_numpatch8_bs16_2048RegionV3"
if not os.path.exists(dir_name):
    raise NameError(f'No directory named {dir_name}')

######################
## Parsing Argument ##
######################
with open(f'{dir_name}/parameter.log', 'r') as f:
    parm_text = f.read()
    matches = re.findall(r'(\w+)\=([^\,\)]+)', parm_text)
params = {}
for match in matches:
    param_name = match[0]
    param_value = match[1]
    params[param_name] = param_value

parser = argparse.ArgumentParser(description='Unsupervised Learning Evaluation')
# Augmentation
parser.add_argument('--num_patches', type=int, default=int(params['num_patches']),
                    help='number of patches used in Augmentation (default: 4), 4 patch number and 64 batch size for 2 30000MiB GPU')
parser.add_argument('--kfold', type=int, default=int(params['kfold']),
                    help='kfold (default: 5)')
parser.add_argument('--transform_type', type=str, default=params['transform_type'].replace("'", ""),
                    help='transform_type parameter for dataset-datasets.py-load_dataset (default: simple)',
                    choices=['simple', 'mask-v1', 'mask-v2', 'region-v1', 'region-v2'])
parser.add_argument('--normalize', type=bool, default=eval(params['normalize']),
                    help='normalize parameter for dataset-datasets.py-load_dataset (default: False)')
# Dataset and Dataloader
parser.add_argument('--bs', type=int, default=int(params['bs']),
                    help='batch size (default: 64), middle batch size')
parser.add_argument('--data', type=str, default=params['data'].replace("'", ""),
                    help='dataset name (default: ips-cell)')
# Model
parser.add_argument('--arch', type=str, default=params['arch'].replace("'", ""),
                    help='network architecture (default: resnet18-cellimage)',
                    choices=['resnet18-cellimage', 'vit-in21k'])     
# Testing setting 
parser.add_argument('--seed', type=int, default=int(params['seed']),
                    help='random seed (default: 3)')
parser.add_argument('--downstream_task', type=str, default=params['downstream_task'].replace("'", ""),
                    help='downstream task (default: None)',
                    choices=['None', '3typesclassification', '2typesclassification', '3typesPCAregression', 
                             '2typesPCAregression', '3typesLevelregression', '2typesLevelregression'])
parser.add_argument('--ips_down', type=bool, default=eval(params['ips_down']),
                    help='whether using iPS cell downstream task (default: False)')
parser.add_argument('--setting_epoch', type=int, default=2,
                    help='setting checking epoch (default: 2)')
parser.add_argument('--qc',type=bool, default=False,
                    help='whether using Quick Check scripts (default: False), if True, remote desktop ssh software is needed')
args = parser.parse_args()
num_patches = args.num_patches
seed = args.seed

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
print(f'Evaluating Using {device}')

#########################
## Dataset Preparation ##
#########################
if args.data == "emt-cell" or args.data == "ips-cell":
    all_dataset = load_dataset(data_name=args.data, num_patch = num_patches, transform_type=args.transform_type, normalize=args.normalize, downstream_task=args.downstream_task, sample_id=True)
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
    # model1 = AutoEncoder(encoder(z_dim=2048, hidden_dim=2048, arch = args.arch), Decoder(latent_dim=2048))
    model1 = AutoEncoder(encoder(z_dim=512, hidden_dim=512, arch = args.arch), Decoder(latent_dim=512))
    model1 = nn.DataParallel(model1)
    model1.load_state_dict(torch.load(f'{dir_name}/save_model1s/{args.seed}/{args.setting_epoch}.pt'))
    model1.to(device)
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
    elif args.downstream_task == '2typesclassification':
        model2 = MLPRegressionModel(input_dim=512, hidden_dim=256, output_dim=2)
    elif args.downstream_task == '3typesPCAregression' or args.downstream_task == '2typesPCAregression':
        model2 = MLPRegressionModel(input_dim=512, hidden_dim=256, output_dim=2)
    elif args.downstream_task == '3typesLevelregression' or args.downstream_task == '2typesLevelregression':
        model2 = MLPRegressionModel(input_dim=512, hidden_dim=256, output_dim=10)
    model2.to(device)
else:
    raise NameError('check setting!')

if __name__ == '__main__':
    if (args.downstream_task == 'None') or ((args.downstream_task in ['3typesclassification','2typesclassification','3typesPCAregression','2typesPCAregression','3typesLevelregression','2typesLevelregression']) and (not args.ips_down)):
        print('MODE 1')
        evaluator = FeatureEvaluator(model1.module.encoder, test_dl, device, num_patches, dir_name, args.setting_epoch, args.downstream_task, args.qc, data_name=args.data)
        evaluator()
    elif args.downstream_task in ['3typesPCAregression','2typesPCAregression','3typesLevelregression','2typesLevelregression']:
        print('MODE 2')
        evaluator = RegressionEvaluator(model1.module.encoder, val_dl, num_patches, dir_name, args.setting_epoch, args.downstream_task, args.qc, model2)
        evaluator()
    else:
        raise NameError('check setting!')