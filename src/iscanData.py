"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import numpy as np
import torch
from PIL import Image
from termcolor import colored

from torchvision import transforms

from utils.collate import collate_custom
from utils.augment2 import Augment, Cutout

import time

class iScanDataset():

    def __init__(self, _iScan, mode):

        self.step = _iScan.step

        self.rawTransform = _iScan.dataset.rawTransform
        self.valTransform = _iScan.dataset.valTransform
        self.augTransform = _iScan.dataset.augTransform

        if mode == 'train':
            self.data = _iScan.dataset.trainSet['data']
            self.labels = _iScan.dataset.trainSet['class']
            if self.step == 'S2':
                self.neighborSet = np.load('%s/S1_topk_train_neighbours.npy' %_iScan.folder)[:, 1: _iScan.config['S2']['num_neighbors'] +1]
                assert(self.neighborSet.shape[0] == len(self.data))

        elif mode == 'valid':
            self.data = _iScan.dataset.validSet['data']
            self.labels = _iScan.dataset.validSet['class']
            if self.step == 'S2':
                self.neighborSet = np.load('%s/S1_topk_val_neighbours.npy' %_iScan.folder)[:, 1: _iScan.config['S2']['num_neighbors'] +1]
                assert(self.neighborSet.shape[0] == len(self.data))

        elif mode == 'eval':
            self.data = _iScan.dataset.trainSet['data']
            self.labels = _iScan.dataset.trainSet['class']
            if self.step == 'S2':
                self.neighborSet = np.load('%s/S1_topk_eval_neighbours.npy' %_iScan.folder)[:, 1: _iScan.config['S2']['num_neighbors'] +1]
                assert(self.neighborSet.shape[0] == len(self.data))
        
    def __len__(self):
        return len(self.data)


class iScanData():

    def getTransforms(self):

        self.dataset.rawTransform = self.getRawTransform()

        if not hasattr(self, 'normMean') or not hasattr(self, 'normStd'):
            self.dataset.valTransform = None
            self.dataset.augTransform = None
            self.normMean, self.normStd = self.getNormValues()
        
        self.dataset.valTransform = self.getValTransform(self.config[self.step])
        self.dataset.augTransform = self.getAugTransform(self.config[self.step])

    def getNormValues(self):

        self.print(colored('Computing image normalization values', 'blue'))

        imgSize = self.config[self.step]['transformation_kwargs']['resize']
        n = len(self.dataset.trainSet['data']) *imgSize *imgSize
        m1, m2 = [], [] # first and second moments

        for i, batch in enumerate(self.getDataLoader('train')):
            numpy_batch = batch['image'].numpy()            
            m1.append(np.sum(numpy_batch, axis = (0, 2, 3)))
            m2.append(np.sum(numpy_batch**2, axis = (0, 2, 3)))

        m1 = np.sum(np.array(m1), axis = 0)
        m2 = np.sum(np.array(m2), axis = 0)
        
        return (m1 /n, np.sqrt(m2 /n -(m1 /n)**2))

    def getDataLoader(self, mode):

        if self.step == 'S1':
            _dataset = self.dataset.AugmentedDataset(self, mode)
        elif self.step == 'S2':
            _dataset = self.dataset.NeighborsDataset(self, mode)
        elif self.step == 'S3':
            _dataset = self.dataset.AugmentedDataset(self, mode)

        if self.config['ddp']:
            train_sampler = torch.utils.data.distributed.DistributedSampler(_dataset, num_replicas = len(_gpu_ids), rank = _rank, shuffle = True)
            return torch.utils.data.DataLoader(_dataset, num_workers = self.config[self.step]['num_workers'],
                batch_size = self.config[self.step]['batch_size'], pin_memory = True, collate_fn = collate_custom,
                drop_last = True, shuffle = False, sampler = train_sampler)
        else:
            return torch.utils.data.DataLoader(_dataset, num_workers = self.config[self.step]['num_workers'], 
            batch_size = self.config[self.step]['batch_size'], pin_memory = True, collate_fn = collate_custom,
            drop_last = (mode == 'train'), shuffle = (mode == 'train'))

    def getRawTransform(self, imgSize = 0):

        if not imgSize:
            imgSize = self.config['image_size']
        return transforms.Compose([transforms.Resize((imgSize, imgSize)), transforms.ToTensor()])

    def getValTransform(self, config):
        
        imgSize = config['transformation_kwargs']['resize']
        return transforms.Compose([
                    transforms.Resize((imgSize, imgSize)),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean = self.normMean, std = self.normStd)
                ])

    def getAugTransform(self, config):
        
        if config['augmentation_strategy'] == 'standard':
            # Standard augmentation strategy
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(**config['augmentation_kwargs']['random_resized_crop']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean = self.normMean, std = self.normStd)
            ])
        elif config['augmentation_strategy'] == 'simclr':
            # Augmentation strategy from the SimCLR paper
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(**config['augmentation_kwargs']['random_resized_crop']),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(**config['augmentation_kwargs']['color_jitter'])
                ], p = config['augmentation_kwargs']['color_jitter_random_apply']['p']),
                transforms.RandomGrayscale(**config['augmentation_kwargs']['random_grayscale']),
                transforms.ToTensor(),
                transforms.Normalize(mean = self.normMean, std = self.normStd)
            ])
        elif config['augmentation_strategy'] == 'scan':
            # Augmentation strategy from our paper 
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(config['augmentation_kwargs']['crop_size']),
                Augment(config['augmentation_kwargs']['num_strong_augs']),
                transforms.ToTensor(),
                transforms.Normalize(mean = self.normMean, std = self.normStd),
                Cutout(
                    n_holes = config['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                    length = config['augmentation_kwargs']['cutout_kwargs']['length'],
                    random = config['augmentation_kwargs']['cutout_kwargs']['random'])])
        else:
            raise ValueError('Invalid augmentation strategy {}'.format(config['augmentation_strategy']))

        return train_transform
