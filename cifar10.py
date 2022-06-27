"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""

import os
import sys
import pickle
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset

from src.iscanData import iScanDataset
from src.iscan import iScan

class Cifar10(Dataset):
        
    '''
    CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>
    '''
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    batch_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    classNames = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    # def __init__(self, config, step):
    def __init__(self, config):

        data = []
        targets = []
        for file_name, checksum in self.batch_list:
            file_path = os.path.join(config['root'], file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)
    
        np.random.seed(seed = config['seed'])
        sample = np.random.choice(np.arange(data.shape[0]), config['trainSize'] +config['validSize'], replace = False)
        train = sample[:config['trainSize']]
        valid = sample[config['trainSize']:]

        # mandatory
        self.trainSet = {'data': data[train], 'class' : targets[train]}
        self.validSet = {'data': data[valid], 'class' : targets[valid]}

    def __len__(self): return len(self.data)

    def getRawImage(self, mode, index):

        if mode == 'valid':
            pilImg = Image.fromarray(self.validSet['data'][index]).convert('RGB')
        else:
            pilImg = Image.fromarray(self.trainSet['data'][index]).convert('RGB')

        rawImg = self.rawTransform(pilImg)
        pilImg.close()

        return rawImg

    class AugmentedDataset(iScanDataset):

        def __getitem__(self, index):

            pilImg = Image.fromarray(self.data[index])

            if self.valTransform is not None:
                original = self.valTransform(pilImg)
            else:
                original = self.rawTransform(pilImg)

            if self.training and self.augTransform is not None:
                augmented = self.augTransform(pilImg)
            else:
                augmented = torch.empty(0, 3)
            
            item = {'image': original, 'image_augmented': augmented, 'target': self.labels[index].item(), 'im_size': pilImg.size, 'index': index}

            pilImg.close()
            
            return item

    class NeighborsDataset(iScanDataset):

        def __getitem__(self, index):

            neighbor_index = np.random.choice(self.neighborSet[index], 1)[0]

            pilImg1 = Image.fromarray(self.data[index])
            pilImg2 = Image.fromarray(self.data[neighbor_index])

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


def load(run = './output/cifar10/202206141105'):

    _iScan = iScan()
    _iScan.readConfig('%s/cifar10.yml' %run, sWriter = True, runlog = run.split('/')[-1])
    _iScan.dataset = Cifar10(_iScan.config)

    return _iScan


if __name__ == '__main__':

    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    config_file = './scan/config/cifar10.yml'
    save = True
    sWriter = True
    run = None

    config_file = './output/cifar10/202206231202/cifar10.yml'
    run = '202206231202'

    _iScan = iScan()
    _iScan.readConfig(config_file, sWriter = sWriter, runlog = run)
    
    stdout = sys.stdout
    with open(os.path.join(_iScan.folder, 'cifar10.log'), 'w') as log_file:

        sys.stdout = log_file

        _iScan.dataset = Cifar10(_iScan.config)
        # _iScan.doStep1(save = save)
        # _iScan.evalStep1()
        # _iScan.topk_cosSim()
        _iScan.doStep2(save = save)
        _iScan.evalStep2()
        _iScan.clusters(step = 'S2')
        # _iScan.doStep3(save = save)
        # _iScan.evalStep3()
        # _iScan.clusters(step = 'S3')

    sys.stdout = stdout

