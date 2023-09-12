'''
Author: Yuxiang Zhang
Date: 2023-08-22
Description:
    FeatureEvaluator for evaluate feature extraction performance using dimentional reduction
    - t-SNE (2D and 3D)
    - PCA
    - KNN (based on t-SNE 2D coordinates)
'''
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FF

from model.models import encoder
from dataset.datasets import load_dataset
import matplotlib.pyplot as plt
from util import chunk_avg

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Evaluation for latant vector
class FeatureEvaluator(object):
    def __init__(self, encoder, test_dl, num_patch, dir_name, epoch):
        self.encoder = encoder
        self.test_dl = test_dl
        self.num_patch = num_patch
        self.dir_name = dir_name + '/fig'
        self.epoch = epoch
    
    def feature_extract(self):
        latent_space = []
        labels = []
        with torch.no_grad():
            for data ,label in self.test_dl:
                data = torch.cat(data, dim=0)
                _, _, z_pre = self.encoder(data, is_test=True)
                z_pre = chunk_avg(z_pre, self.num_patch, normalize=True)
                z_pre = z_pre.detach().cpu().numpy()
                latent_space.append(z_pre)
                labels.extend(label.cpu().numpy())
            latent_space = np.concatenate(latent_space, axis=0)
            print("Latent space shape:", latent_space.shape)
            labels = torch.tensor(labels).reshape(-1,1).numpy().astype(int)
        return  latent_space, labels
    
    def __call__(self):
        print(1)
        self.encoder.eval()
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        latent_space, labels = self.feature_extract()
        cmap = plt.get_cmap('jet', len(np.unique(labels)))
        ## t-SNE
        tsne = TSNE(n_components=2, perplexity=13, random_state=42)# learning_rate=200 (default)
        latent_tsne = tsne.fit_transform(latent_space)
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(ticks=np.arange(len(np.unique(labels))))
        cbar.set_label('Labels')
        plt.title('t-SNE Visualization of Latent Space',fontsize=24)
        plt.xlabel('Dimension 1',fontsize=22)
        plt.ylabel('Dimension 2',fontsize=22)
        plt.savefig(f'{self.dir_name}/tsne_per33_epoch{self.epoch+1}.png',dpi=600)
        ## 3D-tSNE
        tsne = TSNE(n_components=3, perplexity=13, random_state=42)
        latent_tsne = tsne.fit_transform(latent_space)
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], latent_tsne[:, 2], c=labels, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(len(np.unique(labels))))
        cbar.set_label('Labels')
        ax.set_title('t-SNE Visualization of Latent Space', fontsize=24)
        ax.set_xlabel('Dimension 1', fontsize=22)
        ax.set_ylabel('Dimension 2', fontsize=22)
        ax.set_zlabel('Dimension 3', fontsize=22)
        plt.savefig(f'{self.dir_name}/tsne_per33_epoch{self.epoch+1}_3d.png', dpi=600)
        plt.show()
        ## Using PCA
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_space)
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=labels, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(ticks=np.arange(len(np.unique(labels))))
        cbar.set_label('Labels')
        plt.title('PCA Visualization of Latent Space', fontsize=24)
        plt.xlabel('Principal Component 1', fontsize=22)
        plt.ylabel('Principal Component 2', fontsize=22)
        plt.savefig(f'{self.dir_name}/pca_epoch{self.epoch+1}.png', dpi=600)
        ## KNN label
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(latent_tsne.tolist())
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=cluster_labels, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(ticks=np.arange(len(np.unique(cluster_labels))))
        cbar.set_label('Predicted Labels')
        plt.title('t-SNE Visualization of Latent Space',fontsize=24)
        plt.xlabel('Dimension 1',fontsize=22)
        plt.ylabel('Dimension 2',fontsize=22)
        plt.savefig(f'{self.dir_name}/knn_epoch{self.epoch+1}.png',dpi=600)
