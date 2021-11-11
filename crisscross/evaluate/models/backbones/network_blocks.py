# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch
import torch.nn as nn
import numpy as np

class BasicR2P1DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        # src: https://github.com/facebookresearch/AVID-CMA
        super(BasicR2P1DBlock, self).__init__()
        spt_stride = (1, stride[1], stride[2])
        tmp_stride = (stride[0], 1, 1)
        self.spt_conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=spt_stride, padding=(0, 1, 1), bias=False)
        self.spt_bn1 = nn.BatchNorm3d(out_planes)
        self.tmp_conv1 = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=tmp_stride, padding=(1, 0, 0), bias=False)
        self.tmp_bn1 = nn.BatchNorm3d(out_planes)

        self.spt_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.spt_bn2 = nn.BatchNorm3d(out_planes)
        self.tmp_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.out_bn = nn.BatchNorm3d(out_planes)

        self.relu = nn.ReLU(inplace=True)

        if in_planes != out_planes or any([s!=1 for s in stride]):
            self.res = True
            self.res_conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0), bias=False)
        else:
            self.res = False

    def forward(self, x):
        x_main = self.tmp_conv1(self.relu(self.spt_bn1(self.spt_conv1(x))))
        x_main = self.relu(self.tmp_bn1(x_main))
        x_main = self.tmp_conv2(self.relu(self.spt_bn2(self.spt_conv2(x_main))))

        x_res = self.res_conv(x) if self.res else x
        x_out = self.relu(self.out_bn(x_main + x_res))
        return x_out
