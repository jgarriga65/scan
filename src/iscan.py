"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import numpy as np
import math
import shutil

import yaml

from termcolor import colored
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils.utils import mkdir_if_missing
from utils.memory import MemoryBank, fill_memory_bank
from utils.training import cl_train, scan_train, selflabel_train
from utils.losses import SimCLRLoss, SCANLoss, ConfidenceBasedCE
from utils.evaluate import get_predictions

# models
from models.models import ContrastiveModel, ClusteringModel

# +++ DDP 
import subprocess
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
# +++ DDP

from src.iscanData import iScanData
from src.iscanTrain import iScanTrain
from src.iscanEval import iScanEval

class iScan(iScanData, iScanTrain, iScanEval):

    # +++ DDP environment variables
    _rank = 0   #int(os.environ['SLURM_PROCID'])
    _local_rank = 0 #int(os.environ['SLURM_LOCALID'])    
    _size = 1   #int(os.environ['SLURM_NTASKS'])
    _gpu_ids = [0]  #os.environ['SLURM_self.step_GPUS'].split(",")

    def print(self, string):
        if self._rank == 0: print(string)

    def readConfig(self, config_file, sWriter = True, runlog = ''):

        # Retrieve config file
        with open(config_file, 'r') as stream:
            self.config = yaml.safe_load(stream)
        # self.print(colored(self.config, 'red'))

        if not runlog:
            runlog = datetime.now().strftime("%Y%m%d%H%M")

        self.logdir = os.path.join('./log/' , runlog)
        self.folder = os.path.join(self.config['output'], runlog)
        mkdir_if_missing(self.folder)

        # save yml configuration
        if not os.path.exists(os.path.join(self.folder, config_file.split('/')[-1])):
            shutil.copyfile(config_file, os.path.join(self.folder, config_file.split('/')[-1]))

        self.writer = SummaryWriter(self.logdir) if sWriter else None
   
        # CUDNN
        torch.backends.cudnn.benchmark = True

        # DDP
        if self.config['ddp']:
            # if _rank == 0: # print only on master
            #     print(f"Training on {len(_hostnames)} nodes and {_size} processes, master node is {_MASTER_ADDR}")
            #     print(f"Variables for model parallel on one node: {torch.cuda.device_count()} accessible gpus")
            # configure distribution method: define address and port of the master node and initialise
            # communication backend (NCCL)
            # timeout = timedelta(seconds=30), so some timeout problems can be easier to spot
            torch.cuda.set_device(_local_rank)
            dist.init_process_group(backend = "NCCL", timeout = timedelta(seconds = 30), init_method = 'env://', rank = _rank, world_size = _size)

    def getModel(self):

        # Model
        print(colored('Retrieve model', 'blue'))
        if self.step == 'S1':
            self.model = ContrastiveModel(self.config['backbone'], **self.config[self.step]['model_kwargs'])
        elif self.step == 'S2':
            self.model = ClusteringModel(self.config['backbone'], self.config[self.step]['num_classes'], self.config[self.step]['num_heads'])
        elif self.step == 'S3':
            assert(self.config[self.step]['num_heads'] == 1)
            self.model = ClusteringModel(self.config['backbone'], self.config[self.step]['num_classes'], self.config[self.step]['num_heads'])

        # parallelization
        if self.config['ddp']:
            self.model = DDP(self.model, device_ids = [self._local_rank])
        else:
            self.model = torch.nn.DataParallel(self.model)
       
        # Weights
        if self.step == 'S2': 
            # Weights are transfered from contrastive training
            pretrain_path = '%s/S1_model.pth.tar' %self.folder
            if not os.path.exists(pretrain_path):
                raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))
            state = torch.load(pretrain_path, map_location = 'cpu')

            missingKeys = self.model.load_state_dict(state['model'], strict = False)
            assert(set(missingKeys[1]) == {
                'module.contrastive_head.0.weight', 'module.contrastive_head.0.bias', 
                'module.contrastive_head.2.weight', 'module.contrastive_head.2.bias'}
                or set(missingKeys[1]) == {
                'module.contrastive_head.weight', 'module.contrastive_head.bias'})

        elif self.step == 'S3':
            # Weights are transfered from scan 
            # We continue with only the best head (pop all heads first, then copy back the best head)
            pretrain_path = '%s/S2_model.pth.tar' %self.folder
            if not os.path.exists(pretrain_path):
                raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))
            state = torch.load(pretrain_path, map_location = 'cpu')
            model_state = state['model']
            all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
            best_head_weight = model_state['module.cluster_head.%d.weight' %(state['head'])]
            best_head_bias = model_state['module.cluster_head.%d.bias' %(state['head'])]
            # remove model heads
            for k in all_heads: model_state.pop(k)
            # append best head
            model_state['module.cluster_head.0.weight'] = best_head_weight
            model_state['module.cluster_head.0.bias'] = best_head_bias

            missingKeys = self.model.load_state_dict(model_state, strict = True)

        self.print('Model is {}'.format(self.model.__class__.__name__))
        self.print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in self.model.parameters()) / 1e6))

    def getMemoryBank(self, loader = None, device = 'cpu'):

        self.print(colored('Build MemoryBank', 'blue'))
        self.memory_bank = MemoryBank(loader.dataset.__len__(), 
            self.config[self.step]['model_kwargs']['features_dim'],
            self.config[self.step]['num_classes'],
            self.config[self.step]['criterion_kwargs']['temperature'])

        if loader is not None:
            fill_memory_bank(loader, self.model, self.memory_bank)

        if device == 'cpu':
            self.memory_bank.cpu()
        elif device == 'cuda':
            self.memory_bank.cuda()
        else:
            raise ValueError('+++ invalid device for memory_bank !')

    def getCriterion(self):

        self.print(colored('Retrieve criterion', 'blue'))

        if self.config[self.step]['criterion'] == 'simclr':
            criterion = SimCLRLoss(**self.config[self.step]['criterion_kwargs'])

        elif self.config[self.step]['criterion'] == 'scan':            
            criterion = SCANLoss(**self.config[self.step]['criterion_kwargs'])

        elif self.config[self.step]['criterion'] == 'confidence-cross-entropy':

            # Att!! force a low threshold to avoid ValueError: Mask in MaskedCrossEntropyLoss is all zeros. (losses.losses.py)
            # Comment the following line when No-debugging
            # self.config[self.step]['confidence_threshold'] = 0.10
            # criterion = ConfidenceBasedCE(self.config[self.step]['confidence_threshold'], **self.config[self.step]['criterion_kwargs'])

            max_prob, _ = torch.max(get_predictions(self)[0]['probabilities'], dim = 1)
            criterion = ConfidenceBasedCE(max_prob, **self.config[self.step]['criterion_kwargs'])

        else:
            raise ValueError('Invalid criterion {}'.format(self.config[self.step]['criterion']))

        self.criterion = criterion.cuda()
        
        self.print('Criterion is {}'.format(self.criterion.__class__.__name__))
        if self.config[self.step]['criterion'] == 'confidence-cross-entropy':
            print('confidence_quantile %4.2f, threshold %6.4f' %(self.config[self.step]['criterion_kwargs']['confidence_quantile'], self.criterion.threshold))

    def getOptimizer(self):

        self.print(colored('Retrieve optimizer', 'blue'))

        cluster_head_only = (self.step != 'S1' and self.config[self.step]['update_cluster_head_only'])
        if cluster_head_only:
            print(colored('WARNING: SCAN will only update the cluster head', 'red'))
            for name, param in self.model.named_parameters():
                    if 'cluster_head' in name:
                        param.requires_grad = True 
                    else:
                        param.requires_grad = False 
            params = list(filter(lambda param: param.requires_grad, model.parameters()))
            assert(len(params) == 2 * self.config[self.step]['num_heads'])
        else:
            params = self.model.parameters()
                    
        if self.config[self.step]['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(params, **self.config[self.step]['optimizer_kwargs'])
        elif self.config[self.step]['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(params, **self.config[self.step]['optimizer_kwargs'])
        else:
            raise ValueError('Invalid optimizer {}'.format(self.config[self.step]['optimizer']))

        self.print('Optimizer is {}'.format(self.optimizer.__class__.__name__))

    def adjust_learning_rate(self, epoch):

        lr = self.config[self.step]['optimizer_kwargs']['lr']
        if self.config[self.step]['scheduler'] == 'cosine':
            eta_min = lr * (self.config[self.step]['scheduler_kwargs']['lr_decay_rate'] **3)
            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / self.config[self.step]['epochs'])) / 2
        elif self.config[self.step]['scheduler'] == 'steps':
            steps = np.sum(epoch > np.array(self.config[self.step]['scheduler_kwargs']['lr_decay_epochs']))
            if steps > 0:
                lr = lr * (self.config[self.step]['scheduler_kwargs']['lr_decay_rate'] **steps)
        elif self.config[self.step]['scheduler'] == 'constant':
            lr = lr
        else:
            raise ValueError('Invalid learning rate schedule {}'.format(self.config[self.step]['scheduler']))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
  
    def getCheckPoint(self):

        checkPointPath = '%s/%s_checkpoint.pth.tar' % (self.folder, self.step)

        if os.path.exists(checkPointPath):

            self.print(colored('Restart from checkpoint {}'.format(checkPointPath), 'blue'))
            checkpoint = torch.load(checkPointPath, map_location = 'cpu')
            self.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['model'])
            if self.step == 'S2':
                self.best_loss = checkpoint['best_loss']
                self.best_loss_head = checkpoint['best_loss_head']
        else:
            self.start_epoch = 0
            if self.step == 'S2':
                self.best_loss = 1e4
                self.best_loss_head = None

        self.model.cuda()

    def getFinalModel(self):
                        
        # Trained model
        modelPath = '%s/%s_model.pth.tar' % (self.folder, self.step)

        if os.path.exists(modelPath):
            final_state = torch.load(modelPath, map_location = 'cpu')
        else:
            raise ValueError('%s model NOT found ' %modelPath)

        try:
            self.getModel()
            self.model.load_state_dict(final_state['model'])
            if self.step == 'S2':
                self.head = final_state['head']
            self.model.cuda()
            self.getTransforms()

        except:
            raise ValueError('+++ NO training performed')

