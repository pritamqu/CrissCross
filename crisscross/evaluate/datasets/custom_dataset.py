# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, feature_bank):
        super(FeatureDataset, self).__init__()
        self.features = feature_bank['features']
        self.targets = feature_bank['targets']
        self.indexs = feature_bank['indexs']
        
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        feat = self.features[index]
        target = self.targets[index]
        org_idx = self.indexs[index]
        return feat, target, org_idx

