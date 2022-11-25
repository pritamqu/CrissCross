import torch
import numpy as np
from collections import Counter
from bisect import bisect_right
from torch.optim.lr_scheduler import MultiStepLR


class CosineLRAV2(object):
    """ 
    wamup cosine scheduler
    new: pred lr has also a scheduler 
    """
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, iter_per_epoch,
                 vid_base_lr,
                 vid_final_lr,
                 aud_base_lr,
                 aud_final_lr,
                 constant_vid_predictor_lr=False,
                 constant_aud_predictor_lr=False,
                 vid_predictor_lr=None,
                 aud_predictor_lr=None, 
                 ):
        
        # pred_lr_ratio is ratio of base_lr when constant_predictor_lr is False

        self.vid_base_lr    = vid_base_lr
        self.aud_base_lr    = aud_base_lr
        self.vid_final_lr   = vid_final_lr
        self.aud_final_lr   = aud_final_lr
        self.constant_vid_predictor_lr = constant_vid_predictor_lr
        self.constant_aud_predictor_lr = constant_aud_predictor_lr
        
        # sanity check
        if vid_predictor_lr is None:
            vid_predictor_lr = vid_base_lr
        if aud_predictor_lr is None:
            aud_predictor_lr = aud_base_lr        

        self.vid_predictor_lr = vid_predictor_lr
        self.aud_predictor_lr = aud_predictor_lr
        

        warmup_iter = iter_per_epoch * warmup_epochs
        vid_warmup_lr_schedule = np.linspace(warmup_lr, vid_base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        vid_cosine_lr_schedule = vid_final_lr+0.5*(vid_base_lr-vid_final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        
        warmup_iter = iter_per_epoch * warmup_epochs
        aud_warmup_lr_schedule = np.linspace(warmup_lr, aud_base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        aud_cosine_lr_schedule = aud_final_lr+0.5*(aud_base_lr-aud_final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))        
        
        self.vid_lr_schedule = np.concatenate((vid_warmup_lr_schedule, vid_cosine_lr_schedule))
        self.aud_lr_schedule = np.concatenate((aud_warmup_lr_schedule, aud_cosine_lr_schedule))

        if not self.constant_vid_predictor_lr:
            vid_warmup_plr_schedule = np.linspace(warmup_lr, self.vid_predictor_lr, warmup_iter)
            vid_cosine_plr_schedule = vid_cosine_lr_schedule*(self.vid_predictor_lr/self.vid_base_lr) # creating pred cosine lr scheduler from the base cosine scheduler
            self.vid_pred_lr_schedule = np.concatenate((vid_warmup_plr_schedule, vid_cosine_plr_schedule)) 
        if not self.constant_aud_predictor_lr:
            aud_warmup_plr_schedule = np.linspace(warmup_lr, self.aud_predictor_lr, warmup_iter)
            aud_cosine_plr_schedule = aud_cosine_lr_schedule*(self.aud_predictor_lr/self.aud_base_lr) # creating pred cosine lr scheduler from the base cosine scheduler
            self.aud_pred_lr_schedule = np.concatenate((aud_warmup_plr_schedule, aud_cosine_plr_schedule))

        self.optimizer = optimizer
        self.iter = 0
        self.current_vb_lr = self.vid_lr_schedule[self.iter]
        self.current_ab_lr = self.aud_lr_schedule[self.iter]
        self.current_vp_lr = self.vid_predictor_lr if self.constant_vid_predictor_lr else self.vid_pred_lr_schedule[self.iter]
        self.current_ap_lr = self.aud_predictor_lr if self.constant_aud_predictor_lr else self.aud_pred_lr_schedule[self.iter]
        self.current_lr = {
            'video_base_lr': self.current_vb_lr,
            'audio_base_lr': self.current_ab_lr,
            'video_pred_lr': self.current_vp_lr,
            'audio_pred_lr': self.current_ap_lr,           
            }
        
    def step(self):
        for param_group in self.optimizer.param_groups:
            
            if param_group['params']==[]:
                # this filter out param group names w/o any params. 
                pass 

            else:
                                    
                if param_group['name'] == 'aud_predictor':
                    param_group['lr'] = self.aud_predictor_lr if self.constant_aud_predictor_lr else self.aud_pred_lr_schedule[self.iter]
                    
                elif param_group['name'] == 'vid_predictor':
                    param_group['lr'] = self.vid_predictor_lr if self.constant_vid_predictor_lr else self.vid_pred_lr_schedule[self.iter]
                    
                elif param_group['name'] == 'aud_base':
                    param_group['lr'] = self.aud_lr_schedule[self.iter]
    
                elif param_group['name'] == 'vid_base':
                    param_group['lr'] = self.vid_lr_schedule[self.iter] 

                # sanity check incase missed
                else:
                    raise ValueError(f'{param_group} is missing, should not be the case!')
        
        
        self.current_vb_lr = self.vid_lr_schedule[self.iter]
        self.current_ab_lr = self.aud_lr_schedule[self.iter]
        self.current_vp_lr = self.vid_predictor_lr if self.constant_vid_predictor_lr else self.vid_pred_lr_schedule[self.iter]
        self.current_ap_lr = self.aud_predictor_lr if self.constant_aud_predictor_lr else self.aud_pred_lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = {
            'video_base_lr': self.current_vb_lr,
            'audio_base_lr': self.current_ab_lr,
            'video_pred_lr': self.current_vp_lr,
            'audio_pred_lr': self.current_ap_lr,           
            }
        
        return self.current_lr
    
    def get_lr(self):
        return self.current_lr
    
    def state_dict(self):
        return {'iter': self.iter,
                'current_lr': self.current_lr}
    
    def load_state_dict(self, state_dict):
        self.iter = state_dict['iter']
        self.current_lr = state_dict['current_lr']
