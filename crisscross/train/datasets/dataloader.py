# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch
import torch.utils.data as data

def make_dataloader(dataset, 
                     batch_size,
                     use_shuffle,
                     drop_last,
                     num_workers,
                     distributed,
                     pin_memory=True):
    
    if distributed:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=use_shuffle)
        shuffle=False
    else:
        sampler = None
        shuffle=use_shuffle
    
    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=shuffle,
                             sampler=sampler,
                             drop_last=drop_last,)

    return loader