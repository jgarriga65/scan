"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import numpy as np
import math

from termcolor import colored
from datetime import datetime

import torch

from utils.training import cl_train, scan_train, selflabel_train
from utils.memory import fill_memory_bank
from utils.evaluate import contrastive_evaluate, get_predictions, scan_evaluate, hungarian_evaluate

class iScanTrain():

    def doStep1(self, save = True):
        self.doStep(step = 'S1', save = save)

    def doStep2(self, save = True):
        if not os.path.exists('%s/S1_model.pth.tar' %self.folder):
            raise ValueError('+++ run _iSCan.doStep1() first!')
        if not os.path.exists('%s/S1_topk_train_neighbours.npy' %self.folder):
            raise ValueError('+++ run _iSCan.evalStep1() first!')
        self.doStep(step = 'S2', save = save)

    def doStep3(self, save = True):
        if not os.path.exists('%s/S2_model.pth.tar' %self.folder):
            raise ValueError('+++ run _iSCan.doStep2() first!')
        self.doStep(step = 'S3', save = save)

    def doStep(self, step, save = True):
        
        self.step = step
        self.getModel()
        self.getOptimizer()
        self.getCheckPoint()

        # data loaders
        self.getTransforms()
        self.train_loader = self.getDataLoader('train')
        print('Dataset contains {} training samples'.format(self.train_loader.dataset.__len__()))
        self.valid_loader = self.getDataLoader('valid')
        print('Dataset contains {} validation samples'.format(self.valid_loader.dataset.__len__()))       

        if step == 'S1':
            if self.config[self.step]['evaluate']:
                self.eval_loader = self.getDataLoader('eval')
                print('Dataset contains {} evaluation samples'.format(self.eval_loader.dataset.__len__()))
                self.getMemoryBank(loader = self.eval_loader, device = 'cuda')

        self.getCriterion()

        self.ema = None
        if step == 'S3':
            if self.config[self.step]['use_ema']:
                self.ema = EMA(model, alpha = self.config[self.step]['ema_alpha'])
        
        self.print(colored('Starting main loop', 'blue'))
        if step == 'S1':
            self.step1_train(save = save)
            if save: self.step1_saveMemoryBank()
        elif step == 'S2':
            self.step2_train(save = save)
        elif step == 'S3':
            self.step3_train(save = save)
        
    def step1_train(self, save):

        for epoch in range(self.start_epoch, self.config[self.step]['epochs']):

            t = datetime.now()
            self.print(colored('Epoch %d/%d' %(epoch, self.config[self.step]['epochs']), 'yellow'))
            self.print(colored('-'*15, 'yellow'))

            # Adjust lr
            if self.config[self.step]['scheduler'] != 'constant':
                lr = self.adjust_learning_rate(epoch)
                if self.writer is not None:
                    self.writer.add_scalar('step1_lRate', lr, epoch)
                else:
                    self.print('Adjusted learning rate to {:.5f}'.format(lr))

            # Train
            self.print('Train ...')
            cl_train(self, epoch, self._rank)

            # Epoch evaluate
            if self.config[self.step]['evaluate']:
                # Fill memory bank
                self.print('Fill memory bank for kNN...')
                fill_memory_bank(self.eval_loader, self.model, self.memory_bank)
                # Evaluate (To monitor progress - Not for validation)
                self.print('Evaluate ...')
                top1 = contrastive_evaluate(self.valid_loader, self.model, self.memory_bank)
                self.print('Result of kNN evaluation is %.2f' %(top1)) 

            # Checkpoint
            self.print('Checkpoint ...')
            torch.save({'optimizer': self.optimizer.state_dict(), 'model': self.model.state_dict(), 
                        'epoch': epoch + 1}, '%s/%s_checkpoint.pth.tar' % (self.folder, self.step))

            # epoch-time
            self.print('epoch-time ... %s' % str(datetime.now() -t)[:-4])

        # Save final model
        if (save and self.start_epoch < self.config[self.step]['epochs']):
            torch.save({'model': self.model.state_dict()}, '%s/%s_model.pth.tar' % (self.folder, self.step))

    def step1_saveMemoryBank(self, topk = 10):

        # Mine the topk nearest neighbors at the very end
        # topk is the max number of neighbors for config['S2']['num_neighbors']
        print('Mine the nearest neighbors (Top-%d)' %(topk)) 

        # These will be served as input to the SCAN loss.
        loader = self.getDataLoader('train')
        self.getMemoryBank(loader = loader, device = 'cuda')
        self.memory_bank.cpu()
        indices, acc = self.memory_bank.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d faiss nearest neighbors on train set is %.2f' %(topk, 100 *acc))
        np.save('%s/%s_topk_train_neighbours.npy' % (self.folder, self.step), indices)   

        # These will be used for evaluation purposes (for the eval set, same as train set but NOT suffled !!!)
        loader = self.getDataLoader('eval')
        self.getMemoryBank(loader = loader, device = 'cuda')
        self.memory_bank.cpu()
        indices, acc = self.memory_bank.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d faiss nearest neighbors on train set is %.2f' %(topk, 100 *acc))
        np.save('%s/%s_topk_eval_neighbours.npy' % (self.folder, self.step), indices)   

        # These will be used for validation.
        loader = self.getDataLoader('valid')
        self.getMemoryBank(loader = loader, device = 'cuda')
        self.memory_bank.cpu()
        indices, acc = self.memory_bank.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d faiss nearest neighbors on val set is %.2f' %(topk, 100 *acc))
        np.save('%s/%s_topk_val_neighbours.npy' % (self.folder, self.step), indices)   

    def step2_train(self, save):

        for epoch in range(self.start_epoch, self.config[self.step]['epochs']):

            t = datetime.now()
            self.print(colored('Epoch %d/%d' %(epoch+1, self.config[self.step]['epochs']), 'yellow'))
            self.print(colored('-'*15, 'yellow'))
            
            # Adjust lr
            if self.config[self.step]['scheduler'] != 'constant':
                lr = self.adjust_learning_rate(epoch)
                if self.writer is not None:
                    self.writer.add_scalar('step2_lRate', lr, epoch)
                else:
                    self.print('Adjusted learning rate to {:.5f}'.format(lr))
            
            # Train
            self.print('Train ...')
            scan_train(self, epoch)
            
            # Epoch Evaluate (scan_evaluate())
            self.print('Evaluate based on validation set SCAN loss ...')
            predictions = get_predictions(self)
            scan_stats = scan_evaluate(predictions)
            self.print(scan_stats)
            lowest_loss_head = scan_stats['lowest_loss_head']
            lowest_loss = scan_stats['lowest_loss']
            if lowest_loss < self.best_loss:
                self.print('New lowest loss on validation set: %.4f -> %.4f' %(self.best_loss, lowest_loss))
                self.print('Lowest loss head is %d' %(lowest_loss_head))
                self.best_loss = lowest_loss
                self.best_loss_head = lowest_loss_head
                # Save best model
                torch.save({'model': self.model.state_dict(), 'head': self.best_loss_head}, '%s/%s_model.pth.tar' % (self.folder, self.step))
            else:
                self.print('No new lowest loss on validation set: %.4f -> %.4f' %(self.best_loss, lowest_loss))
                self.print('Lowest loss head is %d' %(self.best_loss_head))

            # summaryWriter
            if self.writer is not None:
                self.writer.add_scalar('step2_valLoss', lowest_loss, epoch)

            # Epoch Evaluate (hungarian_evaluate())
            if self.config[self.step]['evaluate']:
                self.print('Evaluate with hungarian matching algorithm ...')
                clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix = False)
                self.print(clustering_stats)     

            # check distribution of clusters
            self.summary(predictions = predictions)
            
            # Checkpoint
            if save:
                self.print('Checkpoint ...')
                torch.save({'optimizer': self.optimizer.state_dict(), 'model': self.model.state_dict(), 
                            'epoch': epoch + 1, 'best_loss': self.best_loss, 'best_loss_head': self.best_loss_head},
                            '%s/%s_checkpoint.pth.tar' % (self.folder, self.step))
            
            # epoch-time
            self.print('epoch-time ... %s' % str(datetime.now() -t)[:-4])

    def step3_train(self, save):
    
        for epoch in range(self.start_epoch, self.config[self.step]['epochs']):
            
            t = datetime.now()
            print(colored('Epoch %d/%d' %(epoch+1, self.config[self.step]['epochs']), 'yellow'))
            print(colored('-'*10, 'yellow'))

            # Adjust lr
            if self.config[self.step]['scheduler'] != 'constant':
                lr = self.adjust_learning_rate(epoch)
                if self.writer is not None:
                    self.writer.add_scalar('step3_lRate', lr, epoch)
                else:
                    print('Adjusted learning rate to {:.5f}'.format(lr))

            # Perform self-labeling 
            print('Train ...')
            selflabel_train(self, epoch)

            # Epoch Evaluate (To monitor progress - Not for validation)
            predictions = get_predictions(self)
            if self.config[self.step]['evaluate']:
                print('Evaluate ...')
                clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False) 
                print(clustering_stats)

            # check distribution of clusters
            self.summary(predictions = predictions)
            
            # Checkpoint
            if save:
                print('Checkpoint ...')
                torch.save({'optimizer': self.optimizer.state_dict(), 'model': self.model.state_dict(), 
                            'epoch': epoch + 1}, '%s/%s_checkpoint.pth.tar' % (self.folder, self.step))
                        
            # epoch-time
            self.print('epoch-time ... %s' % str(datetime.now() -t)[:-4])

        # Save final model
        if (save and self.start_epoch < self.config[self.step]['epochs']):
            torch.save({'model': self.model.state_dict()}, '%s/%s_model.pth.tar' % (self.folder, self.step))
