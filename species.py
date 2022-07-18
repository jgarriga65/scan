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

class Species():

    def __init__(self, config):
        
        fullDataset = []
        for d1 in ['train', 'val', 'test']:
            root = os.path.join(config['root'], d1)
            # for d2 in os.listdir(root):
                # [fullDataset.append(os.path.join(root, d2, f)) for f in os.listdir(os.path.join(root, d2))]
            for d2 in ['unidentifiable', 'aegypti', 'albopictus', 'japonicus-koreicus']:
                [fullDataset.append(os.path.join(root, d2, f)) for f in os.listdir(os.path.join(root, d2))]

        # fullDataset = [os.path.join(config['root'], f) for f in os.listdir(config['root'])]

        # class labels
        labels = [0] *len(fullDataset)

        if config['trainSize'] > 0:

            np.random.seed(seed =config['seed'])
            sample = np.random.choice(fullDataset, config['trainSize'] +config['validSize'], replace = False)

            # mandatory
            self.trainSet = {'data': sample[:config['trainSize']], 'class' : labels[:config['trainSize']]}
            self.validSet = {'data': sample[config['trainSize']:], 'class' : labels[config['trainSize']:]}

        else:

            # this is only for cleaning purposes (!!!)
            self.trainSet = {'data': fullDataset, 'class' : labels}
            self.validSet = {'data': [], 'class' : []}
        
        self.classNames = ['class%s' %str(c) for c in range(config['S1']['num_classes'])]

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

            if self.augTransform is not None:
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

            if self.augTransform is not None:
                anchor = self.augTransform(pilImg1)
                neighbor = self.augTransform(pilImg2)
            elif self.valTransform is not None:
                anchor = self.valTransform(pilImg1)
                neighbor = self.valTransform(pilImg2)
            else:
                anchor = self.rawTransform(pilImg1)
                neighbor = self.rawTransform(pilImg2)
            
            item = {'image': anchor, 'neighbor': neighbor, 'neighborSet': torch.from_numpy(self.neighborSet[index]), 'target': torch.tensor(self.labels[index]), 'index': index}

            pilImg1.close()
            pilImg2.close()
            
            return item


def load(run = './output/species/202206150854'):

    _iScan = iScan()
    _iScan.readConfig('%s/species.yml' %run, sWriter = True, runlog = run.split('/')[-1])
    _iScan.dataset = Species(_iScan.config)

    return _iScan

if __name__ == '__main__':

    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    config_file = './scan/config/species.yml'
    save = True
    sWriter = True

    _iScan = iScan()
    _iScan.readConfig(config_file, sWriter = sWriter)

    stdout = sys.stdout
    with open(os.path.join(_iScan.folder, 'species.log'), 'w') as log_file:

        sys.stdout = log_file

        _iScan.dataset = Species(_iScan.config)
        _iScan.doStep1(save = save)
        _iScan.topk_cosSim()
        _iScan.doStep2(save = save)
        _iScan.clusters(step = 'S2')
        _iScan.doStep3(save = save)
        _iScan.clusters(step = 'S3')

    sys.stdout = stdout

