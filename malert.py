"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset

from src.iscanData import iScanDataset
from src.iscan import iScan

class mAlert():

    # _imgRef =  './datasets/seafile/all_raw'

    # def __init__(self, config):
        
    #     fullDataset = []
    #     for folder in os.listdir(config['root']):
    #         [fullDataset.append(os.path.join(config['root'], folder, f)) for f in os.listdir(os.path.join(config['root'], folder))]

    #     classNames = ['Ae.aegypti', 'Ae.albopictus', 'canNotTell', 'otherSites', 'otherSpecies', 'site', 'notFound']
    #     fullReference = {}
    #     for year in os.listdir(self._imgRef):
    #         with open(os.path.join(self._imgRef, year, 'imgRef.txt')) as f:
    #             imgRef = [row.split(',') for row in f.readlines()][1:]
    #             fullReference[year] = {row[0]: classNames.index(row[7]) for row in imgRef}

    #     np.random.seed(seed =config['seed'])
    #     sample = np.random.choice(fullDataset, config['trainSize'] +config['validSize'], replace = False)

    #     # class labels
    #     labels = []
    #     for row in sample:
    #         year, fKey = row.split('/')[-2:]
    #         try:
    #             labels.append(fullReference[year][str(int(fKey[:-4]))])
    #         except:
    #             labels.append(7)

    #     # mandatory
    #     self.trainSet = {'data': sample[:config['trainSize']], 'class' : labels[:config['trainSize']]}
    #     self.validSet = {'data': sample[config['trainSize']:], 'class' : labels[config['trainSize']:]}
    #     self.classNames = classNames

    def __init__(self, config):
        
        fullDataset = [os.path.join(config['root'], f) for f in os.listdir(config['root'])]

        np.random.seed(seed =config['seed'])
        sample = np.random.choice(fullDataset, config['trainSize'] +config['validSize'], replace = False)

        # class labels
        labels = [0] *len(fullDataset)

        # mandatory
        self.trainSet = {'data': sample[:config['trainSize']], 'class' : labels[:config['trainSize']]}
        self.validSet = {'data': sample[config['trainSize']:], 'class' : labels[config['trainSize']:]}
        # self.classNames = ['class%s' %str(c) for c in range(config['S1']['num_classes'])]
        self.classNames = ['Ae.aegypti', 'Ae.albopictus', 'canNotTell', 'otherSites', 'otherSpecies', 'site', 'notFound']

    def getRawImage(self, mode, index):

        if mode == 'valid':
            pilImg = Image.open(self.validSet['data'][index]).convert('RGB')
        else:
            pilImg = Image.open(self.trainSet['data'][index]).convert('RGB')

        rawImg = self.rawTransform(pilImg)
        pilImg.close()

        return rawImg

    class AugmentedDataset(iScanDataset):

        def __getitem__(self, index):

            pilImg = Image.open(self.data[index]).convert('RGB')

            if self.valTransform is not None:
                original = self.valTransform(pilImg)
            else:
                original = self.rawTransform(pilImg)

            if self.training and self.augTransform is not None:
                augmented = self.augTransform(pilImg)
            else:
                augmented = torch.empty(0, 3)
            
            item = {'image': original, 'image_augmented': augmented, 'target': self.labels[index], 'im_size': pilImg.size, 'index': index}

            pilImg.close()
            
            return item

    class NeighborsDataset(iScanDataset):

        def __getitem__(self, index):

            neighbor_index = np.random.choice(self.neighborSet[index], 1)[0]

            pilImg1 = Image.open(self.data[index]).convert('RGB')
            pilImg2 = Image.open(self.data[neighbor_index]).convert('RGB')

            if self.valTransform is not None:
                anchor = self.valTransform(pilImg1)
            else:
                anchor = self.rawTransform(pilImg1)

            if self.training and self.augTransform is not None:
                neighbor = self.augTransform(pilImg2)
            else:
                neighbor = self.rawTransform(pilImg2)
            
            item = {'image': anchor, 'neighbor': neighbor, 'neighborSet': torch.from_numpy(self.neighborSet[index]), 'target': torch.tensor(self.labels[index]), 'index': index}

            pilImg1.close()
            pilImg2.close()
            
            return item


def load(run = './output/malert/202206150854'):

    _iScan = iScan()
    _iScan.readConfig('%s/malert.yml' %run, sWriter = True, runlog = run.split('/')[-1])
    _iScan.dataset = mAlert(_iScan.config)

    return _iScan

if __name__ == '__main__':

    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    config_file = './scan/config/malert.yml'
    save = True
    sWriter = True

    _iScan = iScan()
    _iScan.readConfig(config_file, sWriter = sWriter)

    stdout = sys.stdout
    with open(os.path.join(_iScan.folder, 'malert.log'), 'w') as log_file:

        sys.stdout = log_file

        _iScan.dataset = mAlert(_iScan.config)
        _iScan.doStep1(save = save)
        _iScan.topk_cosSim()
        _iScan.doStep2(save = save)
        _iScan.clusters(step = 'S2')
        _iScan.doStep3(save = save)
        _iScan.clusters(step = 'S3')

    sys.stdout = stdout

