'''
Author: zyx287
Date: 2023-09-28
Description:
    FeatureEvaluator for classification and 'None' downstream task evaluation
    RegressionEvaluator for regression downstream task evaluation
'''
import os
import tkinter as tk
import numpy as np
import pandas as pd
from PIL import Image

import torch

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from util import chunk_avg, local_chunk_avg

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# For feature evaluation and classification evaluation
class FeatureEvaluator(object):
    def __init__(self, encoder, test_dl, device, num_patch, dir_name, epoch, downstream_task, qc, data_name):
        self.encoder = encoder
        self.test_dl = test_dl
        self.device = device
        self.num_patch = num_patch
        self.dir_name = dir_name + '/fig'
        self.epoch = epoch
        self.downstream_task = downstream_task
        self.qc = qc
        self.data_name = data_name
        if self.qc:# QuickCheck GUI
            if self.data_name == 'emt-cell':
                raise NameError(f'QuickCheck is not supported for {self.data_name} dataset yet (you can mannuly change the filepath in eval.py, then, it will work)')
            self.datalabel = pd.read_csv('/shared/projects/autoencoder/rawdata/2023-08-21.iPSC-RPE/img/label_new_allinone.csv', header=0)
            self.root = tk.Tk()
            self.root.title('On-click PCA QuickCheck')
            self.root.geometry('800x600')
            self.figure = plt.Figure(figsize=(6, 6))
            # Interface
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
    
    # update_plot for QuickCheck
    def update_plot(self, latent_space, labels):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cmap = plt.get_cmap('jet', len(np.unique(labels)))
        sc = ax.scatter(latent_space[:, 0], latent_space[:, 1], c=labels, cmap='jet', alpha=0.7)
        cbar = self.figure.colorbar(sc, ticks=np.arange(len(np.unique(labels))))
        cbar.set_label('Labels')
        ax.set_title('PCA Visualization of Latent Space', fontsize=24)
        ax.set_xlabel('Principal Component 1', fontsize=22)
        ax.set_ylabel('Principal Component 2', fontsize=22)
        self.canvas.draw()
    
    # on_click for QuickCheck
    def on_click(self, event, latent_space, sample_ids):
        if event.xdata is not None and event.ydata is not None:
            # Get the clicked point's index in the latent_space array
            distances = np.sqrt((latent_space[:, 0] - event.xdata) ** 2 +
                                (latent_space[:, 1] - event.ydata) ** 2)
            nearest_point_idx = np.argmin(distances)
            found_time = 0
            # Display the corresponding image
            if nearest_point_idx < len(latent_space):
                # Use nearest_point_idx to get the corresponding image index
                image_index = sample_ids[nearest_point_idx]
                print(type(image_index))
                self.display_image(image_index[0])
            else:
                found_time += 1
                print("{found_time}: Failed, please zoom in and try again!".format(found_time=found_time))

    def display_image(self, idx):
        # Display the image corresponding to the given index
        filename_pd = self.datalabel[self.datalabel['Sample_ID'] == idx]['filename']
        z_index_pd = self.datalabel[self.datalabel['Sample_ID'] == idx]['z_index']
        filename = str(filename_pd.tolist()[0])
        z_index = str(z_index_pd.tolist()[0])

        print(filename, z_index)
        if filename is not None:
            img = Image.open(os.path.join('/shared/projects/autoencoder/rawdata/2023-08-21.iPSC-RPE/img', z_index, filename)).convert('RGB')
            plt.imshow(img)
            plt.title(filename)
            plt.show()
    
    def feature_extract(self):
        global_latent_space = []
        local_latent_space = []
        labels = []
        sample_ids = []
        with torch.no_grad():
            for datapair in self.test_dl:
                if (self.downstream_task == '3typesclassification' or self.downstream_task == '2typesclassification' or self.downstream_task == 'None'):
                    raw_data, label, sample_id = datapair
                else:
                    raise NameError(f'Downstream task {self.downstream_task} is not supported using FeatureEvaluator')
                data = torch.cat(raw_data, dim=0)
                data.to(self.device)
                projection = self.encoder(data)
                avg_local_proj = local_chunk_avg(projection, self.num_patch, normalize=True)
                list_proj = projection.chunk(self.num_patch, dim=0)
                global_proj = list_proj[0]
                # Global
                global_feature = global_proj.detach().cpu().numpy()
                global_latent_space.append(global_feature)
                # Local
                local_feature = avg_local_proj.detach().cpu().numpy()
                local_latent_space.append(local_feature)
                # Label
                labels.extend(label.cpu().numpy())
                sample_ids.extend(sample_id.cpu().numpy())
        global_latent_space = np.concatenate(global_latent_space, axis=0)
        print("Global latent space shape:", global_latent_space.shape)
        local_latent_space = np.concatenate(local_latent_space, axis=0)
        print("Local latent space shape:", local_latent_space.shape)
        labels = torch.tensor(labels).reshape(-1,1).numpy().astype(int)
        sample_ids = torch.tensor(sample_ids).reshape(-1,1).numpy().astype(int)
        return  global_latent_space, local_latent_space, labels, sample_ids
    
    def tsne_ploting(self, latent_space_list, labels, cmap, plot_name):
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_space_list[:, 0], latent_space_list[:, 1], c=labels, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(ticks=np.arange(len(np.unique(labels))))
        cbar.set_label('Labels')
        plt.title('t-SNE Visualization of Latent Space',fontsize=24)
        plt.xlabel('Dimension 1',fontsize=22)
        plt.ylabel('Dimension 2',fontsize=22)
        plt.savefig(f'{plot_name}.png',dpi=600)

    def pca_ploting(self, latent_space_list, labels, cmap, plot_name):
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_space_list[:, 0], latent_space_list[:, 1], c=labels, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(ticks=np.arange(len(np.unique(labels))))
        cbar.set_label('Labels')
        plt.title('PCA Visualization of Latent Space',fontsize=24)
        plt.xlabel('Principal Component 1',fontsize=22)
        plt.ylabel('Principal Component 2',fontsize=22)
        plt.savefig(f'{plot_name}.png',dpi=600)
    
    def __call__(self):
        print('FeatureEvaluator processing')
        self.encoder.eval()
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        global_latent_space, local_latent_space, labels, sample_ids = self.feature_extract()
        cmap = plt.get_cmap('jet', len(np.unique(labels)))
        ## t-SNE
        tsne = TSNE(n_components=2, perplexity=13, random_state=42)# learning_rate=200 (default)
        global_latent_tsne = tsne.fit_transform(global_latent_space)
        local_latent_tsne = tsne.fit_transform(local_latent_space)
        ## PCA
        pca = PCA(n_components=2)
        global_latent_pca = pca.fit_transform(global_latent_space)
        local_latent_pca = pca.fit_transform(local_latent_space)
        ## Plotting
        if not self.qc:
            self.tsne_ploting(global_latent_tsne, labels, cmap, f'{self.dir_name}/global_tsne_per33_epoch{self.epoch}')
            self.tsne_ploting(local_latent_tsne, labels, cmap, f'{self.dir_name}/local_tsne_per33_epoch{self.epoch}')
            self.pca_ploting(global_latent_pca, labels, cmap, f'{self.dir_name}/global_pca_epoch{self.epoch}')
            self.pca_ploting(local_latent_pca, labels, cmap, f'{self.dir_name}/local_pca_epoch{self.epoch}')
            ## KNN label
            labels_all = np.concatenate(labels, axis=0)
            labels_list = labels_all.tolist()
            print("Ground Truth Label: ", labels_list)
            kmeans = KMeans(n_clusters=2, random_state=42)
            global_cluster_labels = kmeans.fit_predict(global_latent_pca.tolist())
            local_cluster_labels = kmeans.fit_predict(local_latent_pca.tolist())
            print('Global predict:', global_cluster_labels.tolist())
            print('Local predict:', local_cluster_labels.tolist())
            sample_ids_all = np.concatenate(sample_ids, axis=0)
            print('sample id: ',sample_ids_all.tolist())
        else:
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvas.mpl_connect('button_press_event',lambda event: self.on_click(event, latent_space=global_latent_pca, sample_ids=sample_ids))
            self.update_plot(global_latent_pca, labels)
            self.root.mainloop()


class RegressionEvaluator(object):
    def __init__(self, encoder, test_dl, num_patch, dir_name, epoch):
        pass

class AERestoring(object):
    def __init__(self, encoder, test_dl, num_patch, dir_name, epoch):
        self.encoder = encoder
        self.test_dl = test_dl
        self.num_patch = num_patch
        self.dir_name = dir_name + '/fig'
        self.epoch = epoch