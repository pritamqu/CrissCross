# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch.nn as nn
import torch


class VideoClassifier(nn.Module):
    "classifier head for full-finetuning"
    def __init__(self, n_classes, feat_dim, l2_norm=False, use_bn=False, use_dropout=False, dropout=0.5):
        super(VideoClassifier, self).__init__()
        self.use_bn = use_bn
        self.l2_norm = l2_norm
        self.use_dropout = use_dropout
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, n_classes)
        self._initialize_weights(self.classifier)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x):
        x = x.squeeze()
        # x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = nn.functional.normalize(x, p=2, dim=-1)
        if self.use_bn:
            x = self.bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        return self.classifier(x)


class VideoFinetune(nn.Module):
    """ This is the wrapper for backbone and classifier head"""
    def __init__(self, backbone, classifier, feat_op='pool'):
        super(VideoFinetune, self).__init__()
        self.feat_op = feat_op
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        with torch.cuda.amp.autocast():
            if self.feat_op =='pool':
                x = self.classifier(self.backbone(x))
            else:
                # extract embeddings from intermediate layer
                x = self.classifier(self.backbone(x, return_embs=True)[self.feat_op])
        return x


class Aud_Wrapper(torch.nn.Module):
    def __init__(self, backbone, feat_op='pool', feat_dim=512, l2_norm=False, use_bn=False):
        """
        wrapper while extracting features for SVM
        change the pooling configuration.
        if feat_op equals to pool, it will take the default pooling layer,
        otherwise pass conv5x to extract features from the final conv activation
        """
        super(Aud_Wrapper, self).__init__()
        self.backbone = backbone
        self.feat_op = feat_op
        self.feat_dim=feat_dim
        self.l2_norm=l2_norm
        self.use_bn=use_bn
        # config src: https://arxiv.org/abs/2003.04298
        pooling_ops = {
                "conv2x": "MaxPool2d(kernel_size=(1, 9), stride=(1,2), padding=0)", # not sure
                "conv3x": "MaxPool2d(kernel_size=(1, 7), stride=(1,2), padding=0)", # not sure
                "conv4x": "MaxPool2d(kernel_size=(1, 5), stride=(1,2), padding=0)", # not sure
                "conv5x": "MaxPool2d(kernel_size=(1, 3), stride=(1,2), padding=0)",
                    }
        if feat_op!='pool':
            self.pooling_layer = eval('torch.nn.'+pooling_ops[feat_op])
        else:
            self.pooling_layer = None
        # feat_dims: [4608]
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)
            # self.bn.weight.data.fill_(1) # doesn't help
            # self.bn.bias.data.zero_() # doesn't help

    def forward(self, x):
        with torch.no_grad():
            if self.pooling_layer is not None:
                x = self.backbone(x, return_embs=True)[self.feat_op]
                x = self.pooling_layer(x)
            else:
                x = self.backbone(x)
            if self.l2_norm:
                x = nn.functional.normalize(x, p=2, dim=-1)
            x = x.view(x.shape[0], -1).contiguous().detach()
        if self.use_bn:
            x = self.bn(x)

        return x

class AudioClassifier(nn.Module):
    "classifier head for finetuning"
    def __init__(self, n_classes, feat_dim, l2_norm=False, use_bn=False, use_dropout=False, dropout=0.5):
        super(AudioClassifier, self).__init__()
        self.use_bn = use_bn
        self.l2_norm = l2_norm
        self.use_dropout = use_dropout
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, n_classes)
        self._initialize_weights(self.classifier)
        self.flatten = nn.Flatten()

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x):
        x = x.squeeze()
        if len(x.shape) > 2:
            x = self.flatten(x)
        if self.l2_norm:
            x = nn.functional.normalize(x, p=2, dim=-1)
        if self.use_bn:
            x = self.bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        return self.classifier(x)

class AudioFCtune(nn.Module):
    """ fc-tune
    use this to evaluate on intermediate layer feature vector"""
    def __init__(self, backbone, classifier, feat_op='conv5x'):
        super(AudioFCtune, self).__init__()
        self.feat_op = feat_op
        self.backbone = backbone
        self.classifier = classifier
        self.flatten = nn.Flatten()
        pooling_ops = {
                "conv2x": "MaxPool2d(kernel_size=(1, 9), stride=(1,2), padding=0)", # not sure
                "conv3x": "MaxPool2d(kernel_size=(1, 7), stride=(1,2), padding=0)", # not sure
                "conv4x": "MaxPool2d(kernel_size=(1, 5), stride=(1,2), padding=0)", # not sure
                "conv5x": "MaxPool2d(kernel_size=(1, 3), stride=(1,2), padding=0)",
                    }
        if feat_op!='pool':
            self.pooling_layer = eval('torch.nn.'+pooling_ops[feat_op])
        else:
            self.pooling_layer = None

    def forward(self, x):
        with torch.no_grad():
            if self.feat_op =='pool':
                x = self.backbone(x)
            else:
                # extract embeddings from intermediate layer
                x = self.backbone(x, return_embs=True)[self.feat_op]
                x = self.pooling_layer(x)
                x = self.flatten(x)
        x = self.classifier(x)
        return x