import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50
from torchmetrics.metric import Metric

from transformers import ViTFeatureExtractor, ViTModel

def getmodel(arch):
    #backbone = resnet18()
    if arch == "resnet18-cifar":
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
  
# Encoder model for feature extraction
class encoder(nn.Module): 
     def __init__(self,z_dim=512,hidden_dim=512, norm_p=2, arch = "resnet18-cifar"):
        '''
        parms:
            z_dim: int, 
        '''
        super().__init__()
        self.arch = arch
        if arch == 'vit-in21k':
            self.feature_extractor = ViTFeatureExtractor(model_name='google/vit-base-patch16-224-in21k')
            # self.feature_extractor = nn.DataParallel(self.feature_extractor)
            self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            # self.vit_model = nn.DataParallel(self.vit_model)
            # self.vit_model.to('cuda')
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
            # print(type(backbone_feature))
            # print(backbone_feature.size())
            # print(backbone_feature[:, 0, :].size())
            feature = self.pre_feature(backbone_feature[:, 0, :])
            projection = self.projection(feature)
            z = F.normalize(projection, p=self.norm_p)
        else:
            backbone_feature = self.backbone(x)# Backbone feature extraction after fc layer
            feature = self.pre_feature(backbone_feature)# Unlinear for feature (Complete the backbone)
            projection = self.projection(feature)# Unlinear for ideal output feature size
            z = F.normalize(projection,p=self.norm_p)# Normalization for finally output
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
        # Transpose convolutional layers for restoring
        self.fc = nn.Linear(latent_dim, 512 * 14 * 14)  # Adjust the hidden dimension if needed
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 512, 14, 14)  # Reshape to match dimensions for convolution
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        output = self.sigmoid(self.deconv4(x))  # Sigmoid for pixel intensity values
        return output
    
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

# Evaluator for evaluating the feature extraction of Encoder
class MLPRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPRegressionModel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32, output_dim),
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class WeightedKNNClassifier(Metric):
    def __init__(
        self,
        k: int = 20,
        T: float = 0.07,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        dist_sync_on_step: bool = False,
    ):
        """Implements the weighted k-NN classifier used for evaluation.
        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
            max_distance_matrix_size (int, optional): maximum number of elements in the
                distance matrix. Defaults to 5e6.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon

        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.
        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    def set_tk(self, T, k):
        self.T = T
        self.k = k
        
    @torch.no_grad()
    def compute(self):
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.
        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """
        
        #print(self.T, self.k)

        train_features = torch.cat(self.train_features)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        chunk_size = min(
            max(1, self.max_distance_matrix_size // num_train_images),
            num_test_images,
        )
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    similarities.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (
                top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
            )  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        self.reset()

        return top1, top5