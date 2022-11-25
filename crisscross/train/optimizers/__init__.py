import torch
from .lr_scheduler import CosineLRAV2
from .larc import LARC

def get_optimizer_av(name, model, lr=1e-3, momentum=0.9, weight_decay=0, betas=(0.9, 0.999)):
    
    # params
    vid_predictor_prefix =  ('module.video_predictor', 'video_predictor') 
    aud_predictor_prefix =  ('module.audio_predictor', 'audio_predictor') 
    
    vid_encoder_prefix =    ('module.video_backbone', 'module.video_projector', 'video_backbone', 'video_projector') 
    aud_encoder_prefix =    ('module.audio_backbone', 'module.audio_projector', 'audio_backbone', 'audio_projector') 
    
    parameters = [
        {
        'name': 'vid_base',
        'params': [param for name, param in model.named_parameters() if name.startswith(vid_encoder_prefix)],
        'lr': lr
        },
        {
        'name': 'vid_predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(vid_predictor_prefix)],
        'lr': lr       
        },
        {
        'name': 'aud_base',
        'params': [param for name, param in model.named_parameters() if name.startswith(aud_encoder_prefix)],
        'lr': lr       
        },
        {
        'name': 'aud_predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(aud_predictor_prefix)],
        'lr': lr        
        }
    ]
       
    # optimizer
    if name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        raise NotImplementedError
        
    return optimizer



