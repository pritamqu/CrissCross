import torch
from .crisscross import *

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}")
    return backbone


def get_model(model_cfg):    

    if model_cfg['name'] == 'CrissCross':
        model =  CrissCross(
            get_backbone(model_cfg['video_backbone']),
            get_backbone(model_cfg['audio_backbone']),
            **model_cfg['kwargs'])    

    else:
        raise NotImplementedError

    return model


def weight_reset(m):
    import torch.nn as nn
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv3d)
        or isinstance(m, nn.ConvTranspose1d)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.ConvTranspose3d)
        or isinstance(m, nn.BatchNorm1d)
        or isinstance(m, nn.BatchNorm2d)
        or isinstance(m, nn.BatchNorm3d)
        or isinstance(m, nn.GroupNorm)
    ):
        m.reset_parameters()


