# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import librosa
import torch
import numpy as np
import random


class AudioPrep(object):
    """ 
    basic audio preparation 
    """
    # src: https://github.com/facebookresearch/AVID-CMA
    
    def __init__(self, 
                 sr=16000,
                 duration=None, 
                 missing_as_zero=False,
                 trim_pad=True, 
                 to_tensor=False):

        self.sr = sr
        self.trim_pad = trim_pad
        self.missing_as_zero = missing_as_zero
        self.to_tensor = to_tensor
        self.duration = duration
        

    def __call__(self, sig):

        num_frames = int(self.duration*self.sr)

        # Check if audio is missing
        if self.missing_as_zero and sig is None:
            sig = np.zeros((1, num_frames), dtype=np.float32)

        # Downmix to mono
        sig = sig.mean(0).astype(np.float32)

        # Trim or pad to constant shape
        if self.trim_pad:
            if sig.shape[0] > num_frames:
                sig = sig[:num_frames]
            elif sig.shape[0] < num_frames:
                n_pad = num_frames - sig.shape[0]
                sig = np.pad(sig, (0, n_pad), mode='constant', constant_values=(0., 0.))

        sig = sig[np.newaxis]
        if self.to_tensor:
            sig = torch.from_numpy(sig)

        return sig

class VolJitter(object):
    def __init__(self, vol=0.1):
        self.vol = vol
        
    def __call__(self, sig):
        sig = sig * random.uniform(1.-self.vol, 1.+self.vol)
        return sig.astype(np.float32)