"""
Authors: 
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import torch
import numpy as np
from termcolor import colored
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

from utils.memory import MemoryBank, fill_memory_bank
from utils.evaluate import contrastive_evaluate, get_predictions, hungarian_evaluate, hungarian_match

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

#from data.malert import mAlert as iScanDataset
#from data.cifar10 import cifar10 as iScanDataset

class iScanEval():

    tagCounter = 0

    def evalStep1(self):

        self.step = 'S1'
        self.getFinalModel()

        # Mine the topk nearest neighbors at the very end (for the training set)
        topk = 10
        print('Mine the nearest neighbors (Top-%d)' %(topk)) 
        loader = self.getDataLoader('eval')
        self.getMemoryBank(loader = loader, device = 'cuda')
        top1 = contrastive_evaluate(loader, self.model, self.memory_bank)
        print('Result of cosSim kNN evaluation is %.2f' %(top1)) 
        self.memory_bank.cpu()
        indices, acc = self.memory_bank.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d faiss nearest neighbors on train set is %.2f' %(topk, 100 *acc))

        # Mine the topk nearest neighbors at the very end (for the validtion set)
        topk = 5
        print('Mine the nearest neighbors (Top-%d)' %(topk)) 
        loader = self.getDataLoader('valid')
        self.getMemoryBank(loader = loader, device = 'cuda')
        top1 = contrastive_evaluate(loader, self.model, self.memory_bank)
        print('Result of cosSim kNN evaluation is %.2f' %(top1)) 
        self.memory_bank.cpu()
        indices, acc = self.memory_bank.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d faiss nearest neighbors on val set is %.2f' %(topk, 100 *acc))

    def evalStep2(self):

        self.step = 'S2'
        self.getFinalModel()

        print(colored('Evaluate model based on SCAN metric', 'blue'))
        self.valid_loader = self.getDataLoader('valid')
        predictions = get_predictions(self)
        clustering_stats = hungarian_evaluate(self.head, predictions, 
                                class_names = self.dataset.classNames, 
                                compute_confusion_matrix = True, 
                                confusion_matrix_file = '%s/%s_confusion_matrix.png' % (self.folder, self.step))
        print(clustering_stats)         

    def evalStep3(self):

        self.step = 'S3'
        self.getFinalModel()

        print(colored('Evaluate model at the end', 'blue'))
        self.valid_loader = self.getDataLoader('valid')
        predictions = get_predictions(self)
        clustering_stats = hungarian_evaluate(0, predictions, 
                                    class_names = self.dataset.classNames,
                                    compute_confusion_matrix = True,
                                    confusion_matrix_file = '%s/%s_confusion_matrix.png' % (self.folder, self.step))
        self.print(clustering_stats)

    def summary (self, step = 'S3', predictions = None):

        if predictions is None:
            self.step = step
            self.getFinalModel()
            self.valid_loader = self.getDataLoader('eval')
            if self.step == 'S2':
                clss, prob, trgt, nghb = get_predictions(self)[0].values()
            else:
                clss, prob, trgt = get_predictions(self)[0].values()

        else:
            if self.step == 'S2':
                clss, prob, trgt, nghb = predictions[0].values()
            else:
                clss, prob, trgt = predictions[0].values()
        
        max_prob, target = torch.max(prob, dim = 1)
        print('+++  -, %5d' %(prob.size()[0]), end = ' ')
        for p in torch.mean(prob, dim = 0).numpy():
            print('%.2f' %np.round(p, 2), end = ' ')
        print()
        print('+++  -, %5d' %(max_prob.size()[0]), end = ' ')
        for thr in np.arange(0.1, 1.0, 0.1):
            print('%.2f' %np.round(torch.sum(max_prob > thr).item() /self.valid_loader.dataset.__len__(), 2), end = ' ')
        print()
        for c in torch.unique(clss):
            max_prob, target = torch.max(prob[clss == c], dim = 1)
            print('+++ %2d, %5d' %(c.item(), max_prob.size()[0]), end = ' ')
            for thr in np.arange(0.1, 1.0, 0.1):
                print('%.2f' %np.round(torch.sum(max_prob > thr).item() /max_prob.size()[0], 2), end = ' ')
            print()

    # +++ neighbor's set visualization +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def topk_faiss(self, topk = 7, sample = 5, mode = 'valid', tag = 'faiss_neighbors'):
        
        self.step = 'S1'
        self.getFinalModel()

        loader = self.getDataLoader(mode)
        self.getMemoryBank(loader = loader, device = 'cuda')

        print('Mine the nearest neighbors (Top-%d)' %(topk)) 
        indices, acc = self.memory_bank.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on %s set is %.2f' %(topk, mode, 100*acc))

        imgSize = self.config[self.step]['augmentation_kwargs']['random_resized_crop']['size']
        batch = torch.zeros((sample *(topk +1), 3, imgSize, imgSize))
        for i, rnd in enumerate(np.random.rand(sample)):
            row = int(round(rnd *loader.dataset.__len__(), 0))
            for j, index in enumerate(indices[row, :]):
                batch[i *(topk +1) +j] = self.dataset.getRawImage(mode, index)

        self.tagCounter += 1
        plotTag = '%s_%s' %(tag, str(self.tagCounter).zfill(3))
        print('+++ writing to %s/%s' %(self.logdir, plotTag))
        self.writer.add_images(plotTag, batch)
        self.writer.close()

    def topk_cosSim(self, topk = 7, sample = 10, tag = 'cosSim_neighbors'):

        self.step = 'S1'
        self.getFinalModel()
        self.getTransforms()
        # mean = torch.from_numpy(self.normMean.reshape(3, 1, 1))
        # std = torch.from_numpy(self.normStd.reshape(3, 1, 1))

        imgSize = min(self.config['image_size'], 128)
        self.dataset.rawTransform = self.getRawTransform(imgSize = imgSize)

        loader = self.getDataLoader('eval')
        self.getMemoryBank(loader = loader, device = 'cuda')

        infImg = torch.zeros((sample, 3, self.config['image_size'], self.config['image_size']))
        infLbl = []
        infIdx = [idx for idx in np.round(np.random.rand(sample) *loader.dataset.__len__(), 0).astype(int)]
        for i, index in enumerate(infIdx):
            item = loader.dataset.__getitem__(index)
            infLbl.append(item['target'])
            infImg[i] = item['image']

        self.model.eval()
        images = infImg.cuda(non_blocking = True)
        output = self.model(images)
        cosSim = torch.matmul(output, self.memory_bank.features.t())
        dist, idxs = cosSim.topk(topk, dim = 1, largest = True, sorted = True)

        neighIdxs = idxs.cpu().numpy()
        neighDist = dist.cpu().detach().numpy()
        neighImgs = torch.zeros((sample *(topk +1), 3, imgSize, imgSize))
        for i in range(sample):
            print('+++ inf:', infIdx[i], infLbl[i])
            print('   topk:', neighIdxs[i])
            print('   lbls:', [loader.dataset.labels[idx] for idx in neighIdxs[i]])
            print('   dist:', neighDist[i])
            # neighImgs[i *(topk +1) +0] = infImg[i]  # *std +mean
            neighImgs[i *(topk +1) +0] = self.dataset.getRawImage('train', infIdx[i])
            for j in range(topk):
                neighImgs[i *(topk +1) +(j +1)] = self.dataset.getRawImage('train', neighIdxs[i][j])

        self.tagCounter += 1
        plotTag = '%s_%s' %(tag, str(self.tagCounter).zfill(3))
        print('+++ writing to %s/%s' %(self.logdir, plotTag))
        self.writer.add_images(plotTag, neighImgs)
        self.writer.close()

    def clusters(self, step = 'S2', mode = 'valid', cluster = -1, samples = 8, tag = 'clustering'):

        self.step = step
        self.getFinalModel()

        imgSize = min(self.config['image_size'], 128)
        self.dataset.rawTransform = self.getRawTransform(imgSize = imgSize)

        self.valid_loader = self.getDataLoader(mode)

        print(colored('Showing clusters, %d samples' %samples, 'blue'))
        pred = get_predictions(self)[0]['predictions']

        if cluster < 0:

            clusterImages = torch.zeros((self.config[self.step]['num_classes'] *samples, 3, imgSize, imgSize))
            for c in range(self.config[self.step]['num_classes']):                
                datasetIdxs = torch.arange(self.valid_loader.dataset.__len__())
                clusterIdxs = datasetIdxs[pred == c]
                print(c, len(clusterIdxs))
                shuffleIdxs = torch.randperm(clusterIdxs.shape[0])
                clusterIdxs = clusterIdxs[shuffleIdxs].view(clusterIdxs.size())
                for i in range(min(samples, clusterIdxs.shape[0])):
                    # print(i, clusterIdxs[i].item(), self.dataset.trainSet['data'][clusterIdxs[i].item()])
                    clusterImages[c *samples +i] = self.dataset.getRawImage(mode, clusterIdxs[i].item())
        
        else:

            clusterImages = torch.zeros((samples, 3, imgSize, imgSize))
            datasetIdxs = torch.arange(self.valid_loader.dataset.__len__())
            clusterIdxs = datasetIdxs[pred == cluster]
            shuffleIdxs = torch.randperm(clusterIdxs.shape[0])
            clusterIdxs = clusterIdxs[shuffleIdxs].view(clusterIdxs.size())
            for i in range(min(samples, clusterIdxs.shape[0])):
                # print(i, clusterIdxs[i].item(), self.dataset.trainSet['data'][clusterIdxs[i].item()])
                clusterImages[i] = self.dataset.getRawImage(mode, clusterIdxs[i].item())

        plotTag = '%s_%s_%s' %('clustering', self.step, mode)
        print('+++ writing to %s/%s' %(self.logdir, plotTag))
        self.writer.add_images(plotTag, clusterImages)
        self.writer.close()


    # +++ UMAP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
    def umap_get(self, nn = 10):

        self.step = 'S1'
        self.getFinalModel()
        self.getMemoryBank(loader = self.getDataLoader('eval'), device = 'cpu')

        print('+++ embedding')
        mapper = UMAP(n_neighbors = nn, min_dist = .1, metric = 'euclidean', init = 'random', low_memory = False)
        self.Y = mapper.fit_transform(self.memory_bank.features.numpy())

    def umap_plot(self, tag = 'UMAP'):

        labels = self.memory_bank.targets.numpy()
        fig, axs = plt.subplots(figsize = (9, 9))
        scatter = axs.scatter(self.Y[:, 0], self.Y[:, 1], c = labels, s = 10, alpha = 0.8)
        handles, labels = scatter.legend_elements()
        slegend = axs.legend(handles, [self.dataset.classNames[i] for i, l in enumerate(labels)], loc = "upper right")
        axs.add_artist(slegend)
        axs.grid(True)
        
        print('+++ writing to %s' %self.logdir)
        self.writer.add_figure(tag, fig)
        self.writer.close()

    # +++ other evaluation utilities +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def chktransform(self, numImgs = 8, tag = 'transform'):

        imgSize = self.config[self.step]['transformation_kwargs']['resize']
        batch = torch.zeros((numImgs*3, 3, imgSize, imgSize))

        self.getTransforms()
        mean = torch.from_numpy(self.normMean.reshape(3, 1, 1))
        std = torch.from_numpy(self.normStd.reshape(3, 1, 1))

        loader = self.getDataLoader('train')
        for i, index in enumerate(np.array(np.random.rand(numImgs) *loader.dataset.__len__(), dtype=np.int)):
            # original image
            batch[i +0] = self.dataset.getRawImage('train', index)
            # transformed/augmented image
            sample = loader.dataset.__getitem__(index)
            batch[i +numImgs] = sample['image'] *std +mean
            batch[i +(2 *numImgs)] = sample['image_augmented'] *std +mean

        self.tagCounter += 1
        plotTag = '%s_%s' %(tag, str(self.tagCounter).zfill(3))
        print('+++ writing to %s/%s' %(self.logdir, plotTag))
        self.writer.add_images(plotTag, batch)
        self.writer.close()

    def clean(self, threshold = 0.999, delete = False, include = []):

        self.step = 'S1'
        self.getFinalModel()
        self.model.eval()
        self.getTransforms()

        imgSize = min(self.config['image_size'], 128)
        self.dataset.rawTransform = self.getRawTransform(imgSize = imgSize)

        loader = self.getDataLoader('eval')
        self.getMemoryBank(loader = loader, device = 'cuda')

        idxList, fNameList = [], []
        for batch in loader:
            images = batch['image'].cuda(non_blocking = True)
            output = self.model(images)
            cosSim = torch.matmul(output, self.memory_bank.features.t())
            dist, idxs = cosSim.topk(2, dim = 1, largest = True, sorted = True)
            if threshold:
                # this mode finds similars up to threshold
                for idx, nghbDst, nghbIdx in zip(batch['index'], dist, idxs):
                    i = idx.item()
                    d = np.round(nghbDst[1].item(), 8)
                    j = nghbIdx[1].item()
                    if d > threshold and j not in idxList and i not in idxList:
                        # print(len(fNameList) +1, i, j, d, self.dataset.trainSet['data'][j].split('/')[-1])
                        print(len(fNameList) +1, i, j, d)
                        print(self.dataset.trainSet['data'][i])
                        print(self.dataset.trainSet['data'][j])
                        idxList.append(i)
                        idxList.append(j)
                        fNameList.append(self.dataset.trainSet['data'][j])
            else:
                # this mode finds real duplicates
                for idx, nghbDst, nghbIdx in zip(batch['index'], dist, idxs):
                    i = idx.item()
                    d = np.round(nghbDst[0].item(), 8)
                    j = nghbIdx[0].item()
                    if j > i:
                        print(len(fNameList) +1, i, j, d)
                        print(self.dataset.trainSet['data'][i])
                        print(self.dataset.trainSet['data'][j])
                        idxList.append(i)
                        idxList.append(j)
                        fNameList.append(self.dataset.trainSet['data'][j])

        if delete:
            for j, fName in enumerate(fNameList):
                if (not len(include) or j +1 in include) and os.path.exists(fName):
                    os.remove(fName)
        else:
            if len(idxList):
                imgList = [self.dataset.getRawImage('eval', idx) for idx in idxList]
                self.writer.add_image('simSet', make_grid(imgList), 0)
                self.writer.close()
        
        return len(fNameList)

    def show_images(self, idxList):

        for idx in idxList:
            print('%d,%s\n' %(idx, self.dataset.trainSet['data'][idx]))

        self.step = 'S1'
        self.getTransforms()

        imgSize = min(self.config['image_size'], 128)
        self.dataset.rawTransform = self.getRawTransform(imgSize = imgSize)
        
        imgList = [self.dataset.getRawImage('train', idx) for idx in idxList]
        self.writer.add_image('imgSet', make_grid(imgList), 0)
        self.writer.close()


    def cMtx(self, step = 'S2', mode = 'valid', norm = None, average = 'macro', plot = False, cmap = 'GnBu'):

        self.step = step
        self.getFinalModel()

        if mode == 'valid':
            true = self.dataset.validSet['class']
        else:
            true = self.dataset.trainSet['class']

        self.valid_loader = self.getDataLoader(mode)
        pred = get_predictions(self)[0]['predictions']
        mtch = hungarian_match(pred, true, len(set(pred)), self.config[self.step]['num_classes'])
        mtch.sort(key = lambda x: x[0])
        pred = [mtch[p][1] for p in pred]

        prc, rec, fsc, sup = precision_recall_fscore_support(true, pred, average = average)
        
        if plot:
            plt.rcParams.update({'font.size': 8})
            ConfusionMatrixDisplay.from_predictions(true, pred, labels = np.arange(self.config[step]['num_classes']), normalize = norm, xticks_rotation = 90.0, values_format = '.2f', cmap = cmap)
            plt.title('prc.%4.2f, rec.%4.2f, fscr.%4.2f' %(prc, rec, fsc), fontsize = 7)
            plt.tight_layout()
            plt.savefig('%s/cMtx.png' %self.folder)
        else:
            m = confusion_matrix(true, pred, labels = np.arange(self.config[step]['num_classes']), normalize = norm)
            if norm is not None: m = np.round(m, 4)
            print('\n +++ cMtx.')
            [print(row) for row in m]
            print('\n +++ prc.%.4f, rec.%.4f, fscr.%.4f' %(prc, rec, fsc))
