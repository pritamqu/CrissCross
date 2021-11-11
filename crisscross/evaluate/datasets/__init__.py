# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os
from datasets.loader.ucf import UCF
from datasets.loader.hmdb import HMDB
from datasets.loader.esc import ESC
import random
import torch


def get_dataset(root, dataset_kwargs, video_transform=None, audio_transform=None, split='train'):
    name = dataset_kwargs['name']
          
    ## action recognition
    if name=='ucf101':
        return UCF(
            DATA_PATH = os.path.join(root, 'UCF-101'),
                 ANNO_PATH = os.path.join(root, 'ucfTrainTestlist'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 return_video=True,
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 return_audio=False,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)

    elif name=='hmdb51':
        return HMDB(
            DATA_PATH = os.path.join(root, 'HMDB-51'),
                 ANNO_PATH = os.path.join(root, 'testTrainMulti_7030_splits'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 return_video=True,
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 return_audio=False,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)
    
    ## sound classification        
    elif name=='esc50':
        return ESC(
            DATA_PATH = os.path.join(root, 'audio'),
                 ANNO_PATH = os.path.join(root, 'meta'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 audio_clip_duration=dataset_kwargs['audio_clip_duration'],
                 audio_fps=dataset_kwargs['audio_fps'],
                 audio_fps_out=dataset_kwargs['audio_fps_out'],
                 audio_transform=audio_transform,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_audio=dataset_kwargs[split]['clips_per_video'],
            )
   
    else:
        raise NotImplementedError
        

def fetch_subset(dataset, size=None):
    if size is None:
        size = len(dataset.classes)
    indices = random.sample(range(len(dataset)), size)
    samples = torch.utils.data.Subset(dataset, indices=indices)
    # samples = subset(dataset, indices=indices)
    return samples

class FetchSubset(torch.utils.data.Subset):

    def __init__(self, dataset, size=None):
        self.dataset = dataset
        if size is None:
            size = len(dataset.classes)
        self.indices = random.sample(range(len(dataset)), size)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.dataset, name)
    