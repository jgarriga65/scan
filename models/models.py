"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
#from models.resnet_cifar import resnet18
#from models.resnet_stl import resnet18

def get_backbone(backbone):

    if backbone == 'resnet18':
        backbone = models.__dict__['resnet18']()
        backbone.fc = nn.Identity()
        return {'backbone': backbone, 'dim': 512}

    elif backbone == 'resnet34':
        backbone = models.__dict__['resnet34']()
        backbone.fc = nn.Identity()
        return {'backbone': backbone, 'dim': 512}

    elif backbone == 'resnet50':
        backbone = models.__dict__['resnet50']()
        backbone.fc = nn.Identity()
        return {'backbone': backbone, 'dim': 2048}

    else:
        raise ValueError('Invalid backbone {}'.format(config['backbone']))

class ContrastiveModel(nn.Module):

    def __init__(self, backbone, head = 'mlp', features_dim = 128):

        super(ContrastiveModel, self).__init__()

        self.backbone, self.backbone_dim = get_backbone(backbone).values()

        self.head = head 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)
        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(),
                    nn.Linear(self.backbone_dim, features_dim))
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):
    
    def __init__(self, backbone, nclusters, nheads = 1):

        super(ClusteringModel, self).__init__()

        self.backbone, self.backbone_dim = get_backbone(backbone).values()

        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)

        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

    def forward(self, x, forward_pass = 'default'):

        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out
