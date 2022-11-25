# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

from tools import Logger, ProgressMeter, AverageMeter, accuracy
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F 
import torch.distributed as dist
from collections import defaultdict
from torch.optim.lr_scheduler import MultiStepLR

########### common stuff

def set_grad(nets, requires_grad=False):
    for param in nets.parameters():
        param.requires_grad = requires_grad
            
########### finetune stuff

def get_optimizer(params, cfg, logger=None):
        
    ## optimizer 
    if cfg['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            params=params,
            lr=cfg['lr']['base_lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            nesterov=False,
        )

    elif cfg['name'] == 'adam':
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfg['lr']['base_lr'],
            weight_decay=cfg['weight_decay'],
            betas=cfg['betas'] if 'betas' in cfg else [0.9, 0.999]
        )

    else:
        raise ValueError('Unknown optimizer.')


    ## lr scheduler 
    if cfg['lr']['name']=='fixed':
        scheduler = MultiStepLR(optimizer, milestones=[cfg['num_epochs']], gamma=1)
    elif cfg['lr']['name']=='multistep':
        scheduler = MultiStepLR(optimizer, milestones=cfg['lr']['milestones'], gamma=cfg['lr']['gamma'])
    else:
        raise NotImplementedError(f"{cfg['lr']['name']} is not yet implemented")
        
    return optimizer, scheduler

def save_checkpoint(args, classifier, optimizer, epoch, name='classifier'):
    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, name + ".pth.tar")
    
    checkpoint = {'optimizer': optimizer.state_dict(), 
                'classifier': classifier.state_dict(), 
                'epoch': epoch + 1}

    torch.save(checkpoint, model_path)
    print(f"Classifier saved to {model_path}")

############## feature extraction

class Feature_Bank(object):
   
    def __init__(self, world_size, distributed, net, logger, print_freq=10, mode='vid', l2_norm=True):

        # mode = vid or aud
        self.mode = mode
        self.world_size = world_size
        self.distributed = distributed
        self.net = net
        self.logger = logger
        self.print_freq = print_freq
        self.l2_norm = l2_norm
        
    @torch.no_grad()
    def fill_memory_bank(self, data_loader):
            
        feature_bank = []
        feature_labels = []
        feature_indexs = []
        self.logger.add_line("Extracting features...")
        phase = 'test_dense' if data_loader.dataset.mode == 'video' else None
        
        for it, sample in enumerate(data_loader):
            if self.mode == 'vid':
                data = sample['frames'] 
            elif self.mode == 'aud':
                data = sample['audio']

            target = sample['label'].cuda(non_blocking=True)
            index = sample['index'].cuda(non_blocking=True)
            
            if phase == 'test_dense':
                batch_size, clips_per_sample = data.shape[0], data.shape[1]
                data = data.flatten(0, 1).contiguous()
                
            feature = self.net(data.cuda(non_blocking=True)).detach()
            if self.l2_norm:
                feature = F.normalize(feature, dim=1) # l2 normalize
            feature = torch.squeeze(feature)
            
            if phase == 'test_dense':
                feature = feature.view(batch_size, clips_per_sample, -1).contiguous()
                
            if self.distributed:
                # create blank tensor
                sub_feature_bank    = [torch.ones_like(feature) for _ in range(self.world_size)]
                sub_labels_bank     = [torch.ones_like(target) for _ in range(self.world_size)]
                sub_index_bank      = [torch.ones_like(index) for _ in range(self.world_size)]
                # gather from all processes
                dist.all_gather(sub_feature_bank, feature)
                dist.all_gather(sub_labels_bank, target)
                dist.all_gather(sub_index_bank, index)
                # concat them 
                sub_feature_bank = torch.cat(sub_feature_bank)
                sub_labels_bank = torch.cat(sub_labels_bank)
                sub_index_bank = torch.cat(sub_index_bank)
                # append to one bank in all processes
                feature_bank.append(sub_feature_bank.contiguous().cpu())
                feature_labels.append(sub_labels_bank.cpu())
                feature_indexs.append(sub_index_bank.cpu())
                
            else:
                
                feature_bank.append(feature.contiguous().cpu())
                feature_labels.append(target.cpu())
                feature_indexs.append(index.cpu())
            
            if it%100==0:
                self.logger.add_line(f'{it} / {len(data_loader)}')
                    
        feature_bank    = torch.cat(feature_bank, dim=0)
        feature_labels  = torch.cat(feature_labels)
        feature_indexs  = torch.cat(feature_indexs)
        
        return feature_bank, feature_labels, feature_indexs
    

def average_features(
    features, 
    labels, 
    indices, 
    logger=None,
    norm_feats=True,
    ):
    
    # src: https://github.com/facebookresearch/selavi/
    
    feat_dict = defaultdict(list)
    label_dict = defaultdict(list)
    for i in range(len(features)):
        if norm_feats:
            v = features[i]
            feat = v / np.sqrt(np.sum(v**2))
        else:
            feat = features[i]
        label = labels[i]
        idx = indices[i]
        feat_dict[idx].append(feat)
        label_dict[idx].append(label)
        print(f'{i} / {len(features)}', end='\r')

    avg_features, avg_indices, avg_labels = [], [], []
    num_features = 0
    for idx in feat_dict:
        stcked_feats = np.stack(feat_dict[idx]).squeeze(axis=0)
        feat = np.mean(stcked_feats, axis=0)
        vid_ix_feat_len = stcked_feats.shape[0]
        num_features += vid_ix_feat_len
        label = label_dict[idx][0]
        avg_features.append(feat)
        avg_indices.append(idx)
        avg_labels.append(label)
    avg_features = np.stack(avg_features, axis=0)
    avg_indices = np.stack(avg_indices, axis=0)
    avg_labels = np.stack(avg_labels, axis=0)

    return avg_features, avg_labels, avg_indices
