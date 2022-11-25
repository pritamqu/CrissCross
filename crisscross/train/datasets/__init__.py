import os
from datasets.loader.kinetics import Kinetics
import random
import torch


def get_dataset(root, dataset_kwargs, video_transform=None, audio_transform=None, split='train'):
    name = dataset_kwargs['name']

    if name=='kinetics400':
        return Kinetics(
            DATA_PATH = os.path.join(root),
                  subset = dataset_kwargs[split]['split'],
                  return_video=dataset_kwargs['return_video'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_audio=dataset_kwargs['return_audio'],
                  audio_clip_duration=dataset_kwargs['audio_clip_duration'],
                  audio_fps=dataset_kwargs['audio_fps'],
                  audio_fps_out=dataset_kwargs['audio_fps_out'],
                  audio_transform=audio_transform,
                  return_labels=False,
                  return_index=False,
                  max_offsync_augm=0,
                  mode=dataset_kwargs[split]['mode'],
                  submode=dataset_kwargs[split]['submode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],)
  
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
    