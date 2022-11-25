# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import librosa
import torch
import numpy as np
from datasets.audiotransforms.torchaudio_transforms import MelSpectrogram as MelSpectrogramTorch # use with power 2

class MelSpectrogramLibrosa(object):
    """Mel spectrogram using librosa.
    """
    # src: https://github.com/nttcslab/byol-a/blob/master/byol_a/dataset.py
    
    def __init__(self, fs=16000, n_fft=1024, shift=160, n_mels=64, fmin=60, fmax=7800):
        self.fs, self.n_fft, self.shift, self.n_mels, self.fmin, self.fmax = fs, n_fft, shift, n_mels, fmin, fmax
        self.mfb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __call__(self, audio):
        X = librosa.stft(np.array(audio), n_fft=self.n_fft, hop_length=self.shift)
        return torch.tensor(np.matmul(self.mfb, np.abs(X)**2 + np.finfo(float).eps))
    
    
