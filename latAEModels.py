#Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
#Licence: Use it however you like, but cite the paper :-)

#Models for Cycle-GAN on encoded data

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lin(c_in, c_out, bn=True, dr=False):  # """Custom convolutional layer for simplicity."""
    layers=[]
    if dr > 0: layers.append(nn.Dropout(dr))
    layers.append(nn.Linear(c_in, c_out, bias=not bn))
    if bn: layers.append(nn.BatchNorm1d(c_out))
    return layers

class G(nn.Module):# """Generator for transfering from mnist to svhn"""
    def __init__(self, cfg):
        super(G, self).__init__()
        lw=cfg["aez"]*np.array([1]+cfg["glay"]+[1])
        self.laleak=cfg["laLeak"]

        lins = []
        for j in range(len(lw)-1):
            dobn = cfg["gben"] and not (cfg["gben"] == 2 and (j + 1 == len(lw) - 1))
            lins+=lin(lw[j],lw[j+1],dobn,cfg["gdrop"][j])
        self.lays=nn.Sequential(*lins)
        
    def forward(self, x):
        for l in self.lays[:-1]:
            x=F.leaky_relu(l(x))

        x=self.lays[-1](x)
        if self.laleak:x=F.leaky_relu(x)
        return x
    
class D(nn.Module):
    def __init__(self, cfg, use_labels=False):
        super(D, self).__init__()
        n_out = 11 if use_labels else 1
        lw = cfg["aez"] * np.array([1] + cfg["dlay"] +[1])
        lw[-1]=n_out
        lins = []
        for j in range(len(lw) - 1):
            dobn=cfg["dben"] and not (cfg["dben"]==2 and (j+1 == len(lw) - 1))
            lins += lin(lw[j], lw[j + 1],dobn,cfg["ddrop"][j])

        self.lays = nn.Sequential(*lins)

    def forward(self, x):
        for l in self.lays[:-1]:
            x = F.leaky_relu(l(x))
        x = self.lays[-1](x)
        return x


class LinCl(nn.Module):
    def __init__(self, cfg):
        super(LinCl, self).__init__()
        lw = [cfg["aez"]]+list(cfg["aez"]*np.array(cfg["cllay"]))+[cfg["ds"][1]]

        lins = []
        for j in range(len(lw) - 1):
            dobn = cfg["clben"] and not (cfg["clben"] == 2 and (j + 1 == len(lw) - 1))
            lins += lin(lw[j], lw[j + 1], dobn, cfg["cldrop"][j])
        self.lays = nn.Sequential(*lins)

    def forward(self, x):
        for l in self.lays[:-1]:
            x = F.leaky_relu(l(x))
        x = self.lays[-1](x)
        return x

