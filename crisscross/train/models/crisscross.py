import torch
import torch.nn as nn
import torch.nn.functional as F 
from .backbones import R2Plus1D, ResNet

def D(p, z): 
    # src: https://github.com/PatrickHua/SimSiam
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

## heads
class projection_MLP(nn.Module):
    # src: https://github.com/PatrickHua/SimSiam
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

class prediction_MLP(nn.Module):
    # src: https://github.com/PatrickHua/SimSiam
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class CrissCross(nn.Module):

    def __init__(self, 
                 video_backbone=R2Plus1D, 
                 audio_backbone=ResNet, 
                 video_depth=18,
                 audio_depth=18,
                 vid_proj_layers=2,
                 aud_proj_layers=2,
                 joint=True,
                 cm=True,
                 ctcm=True,
                 vid_coeff=1,aud_coeff=1,cm_coeff=1, ct_coeff=1,
                 pre_norm=False, 
                 ):
        super().__init__()
        
        self.video_backbone = video_backbone(video_depth)        
        self.video_flatten = nn.Flatten()
        self.video_projector = projection_MLP(self.video_backbone.__dict__['output_dim'])
        self.video_encoder = nn.Sequential(self.video_backbone, self.video_flatten, self.video_projector)
        self.video_predictor = prediction_MLP()
        self.video_projector.set_layers(vid_proj_layers)

        self.audio_backbone = audio_backbone(audio_depth)
        self.audio_flatten = nn.Flatten()
        self.audio_projector = projection_MLP(self.audio_backbone.__dict__['output_dim'])
        self.audio_encoder = nn.Sequential(self.audio_backbone, self.audio_flatten, self.audio_projector)
        self.audio_predictor = prediction_MLP()
        self.audio_projector.set_layers(aud_proj_layers)
        
        self.joint, self.cm, self.ctcm = joint, cm, ctcm
        self.num_loss = 0
        self.num_loss += 2 if self.joint else 0
        self.num_loss += 2 if self.cm else 0
        self.num_loss += 2 if self.ctcm else 0
        
        self.vid_coeff, self.aud_coeff, self.cm_coeff, self.ct_coeff = vid_coeff, aud_coeff, cm_coeff, ct_coeff
        self.pre_norm = pre_norm

    
    def forward(self, v1, v2, a1, a2):
                
        with torch.cuda.amp.autocast():
            
            # video
            fv, hv = self.video_encoder, self.video_predictor
            # video embeddings from the encoder (backbone+projector)
            zv1, zv2 = fv(v1), fv(v2)
            if self.pre_norm:
                # normalize first
                zv1 = F.normalize(zv1, dim=1)
                zv2 = F.normalize(zv2, dim=1)
            # feed embeddings to the predictor
            pv1, pv2 = hv(zv1), hv(zv2)
            
            # audio
            fa, ha = self.audio_encoder, self.audio_predictor
            # audio embeddings from the encoder (backbone+projector)
            za1, za2 = fa(a1), fa(a2)
            if self.pre_norm:
                # normalize first
                za1 = F.normalize(za1, dim=1)
                za2 = F.normalize(za2, dim=1)
            # feed embeddings to the predictor
            pa1, pa2 = ha(za1), ha(za2)
            
            # loss calculations
            Lv1v2, La1a2, Lv1a2, La1v2, Lv1a1, Lv2a2 = 0, 0, 0, 0, 0, 0
            if self.joint:
                # though normalized feature is not required here, but probably that's okay
                # v1-v2 - joint
                Lv1v2 = D(pv1, zv2) / 2 + D(pv2, zv1) / 2
                # a1-a2 - joint
                La1a2 = D(pa1, za2) / 2 + D(pa2, za1) / 2
                
            if self.ctcm:
                # v1-a2 - cross time cross modal
                Lv1a2 = D(pv1, za2) / 2 + D(pa2, zv1) / 2
                # a1-v2 - cross time cross modal
                La1v2 = D(pa1, zv2) / 2 + D(pv2, za1) / 2
                
            if self.cm:
                # v1-a1 - cross modal
                Lv1a1 = D(pv1, za1) / 2 + D(pa1, zv1) / 2
                # v2-a2 - cross modal
                Lv2a2 = D(pv2, za2) / 2 + D(pa2, zv2) / 2
            
            L = ((Lv1v2 * self.vid_coeff + La1a2 * self.aud_coeff) + \
                    (Lv1a2 + La1v2) * self.ct_coeff + \
                        (Lv1a1 + Lv2a2) * self.cm_coeff)/self.num_loss
                            
        return {
            'loss': L, 
            'subloss': {
                'Lv1v2': Lv1v2,
                'La1a2': La1a2,
                'Lv1a2': Lv1a2,
                'La1v2': La1v2, 
                'Lv1a1': Lv1a1,
                'Lv2a2': Lv2a2,
                }
            }




