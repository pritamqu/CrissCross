# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

def get_vid_aug(name='standard', crop_size=224, num_frames=8, mode='train', aug_kwargs=None):

    from .video_augmentations import StandardTransforms, StrongTransforms

    if name == 'standard':
        augmentation = StandardTransforms(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs)
        
    elif name == 'strong':
        augmentation = StrongTransforms(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs)        
    else:
        raise NotImplementedError

    return augmentation


def get_aud_aug(name='standard', audio_fps=16000, n_fft=1024, n_mels=80, duration=2, hop_length=160, mode='train', aug_kwargs=None):
    
    from .audio_augmentations import StandardAug, StrongAug

    if name == 'standard':
        augmentation = StandardAug(
            mode=mode,
            audio_fps=audio_fps,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            duration=duration,
            **aug_kwargs)
                    
    elif name == 'strong':
        augmentation = StrongAug(
            mode=mode,
            audio_fps=audio_fps,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            duration=duration,
            **aug_kwargs) 

    else:
        raise NotImplementedError

    return augmentation
