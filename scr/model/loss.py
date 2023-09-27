'''
Author: zyx287
Date: 2023-08-03
Description:
    Loss function for training SSL on cell images
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature Loss
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
                
        return -z_sim, z_sim_out
    
############################
## Option Loss (Not used) ##
############################

# No need for using negative samples
class contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__
        pass
    def forward(self, x, labels):
        #positive logit is always the first element
        '''
        x[:, 0]: Logit for positive sample
        x[:, 1:]: logit for all negetive samples
        -x[:, 0] + torch.log(torch.sum(torch.exp(x[:, 1:]), dim=-1)): logit for positive - logit for all negetive
        L = -x[0] + log(Σexp(x[1:]))
        '''
        loss = -x[:,0] + torch.log(torch.sum(torch.exp(x[:,1:]), dim=-1))
        return torch.mean(loss)

# Not suitable for float64 vector(?
class mutual_loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, global_vector, infered_vector):
        '''
        global_vector: global feature vector
        infered_vector: infered feature vector
        '''
        joint_prob = np.zeros((np.max(global_vector)+1, np.max(infered_vector)+1))
        for i in range(len(global_vector)):
            joint_prob[global_vector[i], infered_vector[i]] += 1
        joint_prob = joint_prob/len(global_vector)
        joint_prob = torch.tensor(joint_prob, dtype=torch.float32)
        # Marginal probability
        marginal_global = torch.sum(joint_prob, dim=1)
        marginal_infered = torch.sum(joint_prob, dim=0)
        # Mutual information
        mutual_info = 0
        for i in range(len(global_vector)):
            p_global = marginal_global[global_vector[i]]
            p_infered = marginal_infered[infered_vector[i]]
            p_joint = joint_prob[global_vector[i], infered_vector[i]]
            if p_joint > 0 and p_global > 0 and p_infered > 0:
                mutual_info += p_joint * torch.log(p_joint/(p_global*p_infered))
        return mutual_info