"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter


def cl_train(_iScan, epoch, _rank):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(_iScan.train_loader),
        [losses],
        prefix="dev_{} Epoch: [{}]".format(_rank, epoch))

    _iScan.model.train()

    avgLoss = 0
    for i, batch in enumerate(_iScan.train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = _iScan.model(input_).view(b, 2, -1)
        loss = _iScan.criterion(output)
        losses.update(loss.item())

        _iScan.optimizer.zero_grad()
        loss.backward()
        _iScan.optimizer.step()

        avgLoss += loss.item()

        if i % _iScan.config[_iScan.step]['batch_info'] == 0:
            progress.display(i)

    # summaryWriter
    if _iScan.writer is not None:
        _iScan.writer.add_scalar('step1_loss', avgLoss /(i +1), epoch)


def scan_train(_iScan, epoch, update_cluster_head_only = False):
    """ 
    Train w/ SCAN-Loss
    """
    print('Epoch' +' ' *13 +'TotalLoss' +' ' *22 +'ConsistencyLoss' +' ' *17 +'Entropy')
    total_losses = AverageMeter('', ':.4e')
    consistency_losses = AverageMeter('', ':.4e')
    entropy_losses = AverageMeter('', ':.4e')
    progress = ProgressMeter(len(_iScan.train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix = "[{}]".format(epoch +1))

    if _iScan.config[_iScan.step]['update_cluster_head_only']:
        _iScan.model.eval() # No need to update BN
    else:
        _iScan.model.train() # Update BN

    step2_loss = .0
    for i, batch in enumerate(_iScan.train_loader):
        # Forward pass
        anchors = batch['image'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
       
        if _iScan.config[_iScan.step]['update_cluster_head_only']: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = _iScan.model(anchors, forward_pass = 'backbone')
                neighbors_features = _iScan.model(neighbors, forward_pass = 'backbone')
            anchors_output = _iScan.model(anchors_features, forward_pass = 'head')
            neighbors_output = _iScan.model(neighbors_features, forward_pass = 'head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = _iScan.model(anchors)
            neighbors_output = _iScan.model(neighbors)     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = _iScan.criterion(anchors_output_subhead, neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))
        step2_loss += total_loss.item()

        _iScan.optimizer.zero_grad()
        total_loss.backward()
        _iScan.optimizer.step()

        if i % _iScan.config[_iScan.step]['batch_info'] == 0:
            progress.display(i)

    # summaryWriter
    if _iScan.writer is not None:
        _iScan.writer.add_scalar('step2_trainLoss', step2_loss /(i +1), epoch)


def selflabel_train(_iScan, epoch):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    confidents = AverageMeter('Conf./batch', ':.0f')
    progress = ProgressMeter(len(_iScan.train_loader), [losses, confidents],
                                prefix="Epoch: [{}]".format(epoch))
    _iScan.model.train()

    step3_loss = .0
    for i, batch in enumerate(_iScan.train_loader):
            
        images = batch['image'].cuda(non_blocking = True)
        images_augmented = batch['image_augmented'].cuda(non_blocking = True)

        with torch.no_grad(): 
            output = _iScan.model(images)[0]
        output_augmented = _iScan.model(images_augmented)[0]

        loss, conf = _iScan.criterion(output, output_augmented)
        losses.update(loss.item())
        confidents.update(conf)
        step3_loss += loss.item()
        
        _iScan.optimizer.zero_grad()
        loss.backward()
        _iScan.optimizer.step()

         # Apply EMA to update the weights of the network
        if _iScan.ema is not None:
            _iScan.ema.update_params(_iScan.model)
            _iScan.ema.apply_shadow(_iScan.model)
        
        if i % _iScan.config[_iScan.step]['batch_info'] == 0:
            progress.display(i)

    # summaryWriter
    if _iScan.writer is not None:
        _iScan.writer.add_scalar('step3_loss', step3_loss /(i +1), epoch)
