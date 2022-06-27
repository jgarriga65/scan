"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import math
import numpy as np
import torch


def get_train_dataset(p, transform, split = None):

    if p['train_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=True, transform=transform, download=True)
    elif p['train_db_name'] == 'cifar-20':
        from data.cifar import CIFAR20
        dataset = CIFAR20(train=True, transform=transform, download=True)
    elif p['train_db_name'] == 'stl-10':
        from data.stl import STL10
        dataset = STL10(split=split, transform=transform, download=True)
    elif p['train_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='train', transform=transform)
    elif p['train_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['train_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='train', transform=transform)
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    return dataset


def get_val_dataset(p, transform = None):

    if p['val_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=False, transform=transform, download=True)
    elif p['val_db_name'] == 'cifar-20':
        from data.cifar import CIFAR20
        dataset = CIFAR20(train=False, transform=transform, download=True)
    elif p['val_db_name'] == 'stl-10':
        from data.stl import STL10
        dataset = STL10(split='test', transform=transform, download=True)
    elif p['val_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='val', transform=transform)
    elif p['val_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['val_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='val', transform=transform)
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))

    return dataset


