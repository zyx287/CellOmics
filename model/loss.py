'''
Author: Yuxiang Zhang
Date: 2023-08-24
Description:
    Loss function for training SSL on cell images
    - TCR
    - Similarity Loss
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Contrastive loss for training
class contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__
        pass
    def forward(self, x, labels):
        '''
        x[:, 0]: Logit for first sample
        x[:, 1:]: logit for other samples
        L = -x[0] + log(Σexp(x[1:]))
        '''
        loss = -x[:,0] + torch.log(torch.sum(torch.exp(x[:,1:]), dim=-1))
        return torch.mean(loss)

# TCR for monitoring the collapsing of representation
class TotalCodingRate(nn.Module):
    '''
    Mostly copy-paste from https://arxiv.org/abs/2304.03977
    '''
    def __init__(self, eps=0.01):
        super(TotalCodingRate,self).__init__()
        self.eps = eps
    
    def compute_discrimn_loss(self, W):
        # Discriminative Loss
        '''
        A soft-constrained regularization of covariance term in VICReg
        '''
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(scalar * torch.matmul(W, W.t()) + I)
        return logdet / 2.0
    
    def forward(self, X):
        return self.compute_discrimn_loss(X.T)

# Cosine similarity loss for feature loss
class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
        # -z_sim is the loss(to be minimized)        
        return -z_sim, z_sim_out