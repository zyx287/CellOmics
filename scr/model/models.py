'''
Author: Yuxiang Zhang
Date: 2023-09-04
Description:
    CellOmics Model Arch
    - Encoder Backbone
      - ResNet18
      - Vision Transformer
    - Transpose Convolutional Decoder
    - AutoEncoder
'''
import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50
from torchmetrics.metric import Metric

from transformers import ViTFeatureExtractor, ViTModel

def getmodel(arch):
    '''
    Only used for ResNet backbone
    '''
    if arch == "resnet18-cellimage":
        '''
        Smaller conv1 and no maxpool for small image feature extraction
        '''
        backbone = resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        return backbone, 512
    elif arch == "resnet18-cell":
        backbone = resnet18()
        backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        backbone.fc = nn.Identity()
        return backbone, 512
    else:
        raise NameError("{} not found in network architecture".format(arch))
  
# Encoder
class encoder(nn.Module): 
     def __init__(self,z_dim=512,hidden_dim=512, norm_p=2, arch = "resnet18-cellimage"):
        '''
        parms:
            z_dim: int, the output dimension of encoder
            hidden_dim: int, the hidden dimension of encoder
            norm_p: int, the norm of output vector
        '''
        super().__init__()
        self.arch = arch
        if arch == 'vit-in21k':
            self.feature_extractor = ViTFeatureExtractor(model_name='google/vit-base-patch16-224-in21k')
            self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.norm_p = norm_p
            self.pre_feature = nn.Sequential(
                nn.Linear(self.vit_model.config.hidden_size, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) 
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.BatchNorm1d(hidden_dim), 
                nn.ReLU(), 
                nn.Linear(hidden_dim, z_dim)
            )
        else:
            backbone, feature_dim = getmodel(arch)
            self.backbone = backbone
            self.norm_p = norm_p
            self.pre_feature = nn.Sequential(nn.Linear(feature_dim,hidden_dim),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU()
                                            )
            self.projection = nn.Sequential(nn.Linear(hidden_dim,hidden_dim), 
                                            nn.BatchNorm1d(hidden_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(hidden_dim,z_dim)
                                            )
          
     def forward(self, x, is_test = False):
        '''
        parm:
            is_test: determine the output format, False when training
        '''
        if self.arch == 'vit-in21k':
            inputs = self.feature_extractor(images=x, return_tensors='pt')
            inputs.to(x.device)
            vit_outputs = self.vit_model(**inputs)
            backbone_feature = vit_outputs.last_hidden_state
            print(type(backbone_feature))
            print(backbone_feature.size())
            print(backbone_feature[:, 0, :].size())
            feature = self.pre_feature(backbone_feature[:, 0, :])
            projection = self.projection(feature)
            z = F.normalize(projection, p=self.norm_p)
        else:
            backbone_feature = self.backbone(x)
            feature = self.pre_feature(backbone_feature)
            projection = self.projection(feature)
            z = F.normalize(projection,p=self.norm_p)
        # Output format
        if is_test:
            return feature, projection, z
        else:
            return z

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # Vector dimension after projection
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 512 * 14 * 14)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 512, 14, 14)# Match dimensions for convolution
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        output = self.sigmoid(self.deconv4(x))
        return output
    
# CellOmics Model Arch
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, Para=True):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = decoder
        self.decoder = nn.DataParallel(self.decoder)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x