import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class FocalLoss(nn.Module):
    def __init__(self, numlabel, mode, gamma=2):
        super(FocalLoss,self).__init__()
        a1=torch.log(torch.tensor(numlabel))
        a2=a1/sum(a1)
        alpha=1/a2/torch.mean((1/a2))
        
        self.alpha=alpha
        self.gamma=gamma
        self.mode = mode

    def forward(self, input, target, gt):
        # self.alpha=self.alpha.to(input.device)
        # important to add reduction='none' to keep per-batch-item loss
        if self.mode=='sgdet':
            with autocast(enabled=False):
                loss=nn.functional.binary_cross_entropy(\
                    input.float(), target.float(), reduction='none').sum(1)
        else:
            loss = nn.functional.binary_cross_entropy(input, target, reduction='none').sum(1)
        pt = torch.exp(-loss)
        realalpha=[np.mean([self.alpha[g-1] for g in agt]) for agt in gt]
        realalpha=torch.tensor(realalpha).to(input.device)

        # mean over the batch
        focal_loss = (realalpha * (1-pt)**self.gamma * loss).mean()
        return focal_loss

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_timing_signal_1d(length,channels,min_timescale=1.0,max_timescale=1.0e4,start_index=0):
    position=torch.arange(length).float()+start_index
    num_timescales = float(channels // 2)
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (num_timescales - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).float() * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1)
    signal = torch.nn.functional.pad(signal, (0, channels%2,0,0))
    return signal.view([length, channels])