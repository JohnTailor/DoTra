#Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
#Licence: Use it however you like, but cite the paper :-)

#Classifier models

import numpy as np
import torch
import torch.nn as nn

class BBlock(nn.Module):
    def __init__(self, in_planes, planes,ker=3,down=True,pad=1):
        super(BBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=ker, stride=1, padding=pad, bias=False)
        self.bn=nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d((2, 2), stride=2) if down else nn.Identity()

    def forward(self, x):
        out=self.conv1(x) #out = self.ident(out)
        out= self.bn(out)        #out = self.identBefR(out)
        out = self.relu(out)        #out = self.identBefS(out)
        out = self.mp(out)
        return out


class worNet(nn.Module):
    def __init__(self, cfg):
        super(worNet, self).__init__()
        cf= cfg["clcfg"]
        tr = lambda x: max(1, int(np.round(x * cfg["netSi"])))
        #self.addN=cfg["addN"]        self.oneh = cfg["onehot"]
        self.in_channels = cfg["imCh"]
        in_channels = self.in_channels
        self.is11 = 1 if "11" in cf["netT"] else 0
        #chans = [in_channels, 32, 64, 64,  128, 128, 256, 256, 512, 512] if self.is11 else [in_channels, 32, 64, 128, 256, 512]
        chans = [in_channels, 64, 64, 64, 128, 128, 256, 256, 512, 512] if self.is11 else [in_channels, 64, 64, 128, 256, 512]

        i=-1
        def getConv(ker=cfg["ker"], down=True):
            nonlocal i
            i+=1 #return nn.Sequential(*[nn.Conv2d(in_channels=inoffs[i]+ (tr(chans[i]) if i>0 else chans[i]) , out_channels=tr(chans[i+1]), kernel_size=(ker, ker), padding=ker > 1), nn.BatchNorm2d(tr(chans[i+1])), relu] + ([mp] if down else []))
            return BBlock((tr(chans[i]) if i>0 else chans[i]),tr(chans[i+1]), ker=ker,down=down,pad=(ker-1)//2)#inoffs[i]+

        #if self.is11: self.conv0a = nn.Identity()
        self.conv0 = getConv()
        if self.is11: self.conv1a = getConv(down=False)
        self.conv1 = getConv()
        if self.is11: self.conv2a = getConv( down=False)
        self.conv2 = getConv()
        if self.is11: self.conv3a = getConv(ker=3, down=False)
        self.conv3 = getConv(ker=3)
        if self.is11: self.conv4a = getConv( down=False,ker=3)
        self.conv4 = getConv(ker=3)

        self.allays = [self.conv0, self.conv1, self.conv2, self.conv3, self.conv4]
        if self.is11: self.allays = [self.conv0,self.conv1a,self.conv1, self.conv2a,self.conv2, self.conv3a,self.conv3, self.conv4a, self.conv4]
        i, ker = -1, 1
        self.avgpool = nn.AdaptiveMaxPool2d((1,1))
        self.dropout = nn.Dropout(0.5) if cfg["drop"] else nn.Identity()
        self.pred = nn.Linear(tr(512),tr(128) ) if cfg["twoLin"] else nn.Identity()
        self.pred2 = nn.Linear(tr(128),cfg["ds"][1]) if cfg["twoLin"] else nn.Linear(tr(512),cfg["ds"][1])
        #self.k=0

    def forward(self, x):
        # import imgutil as imgu        # print(np.sum(np.abs(x.cpu().numpy())))        # imgu.makeAndStore(x.cpu().numpy(),x.cpu().numpy(),"Img",str(self.k)+".png")        # self.k+=1        # self.k=self.k%10
        for il,l in enumerate(self.allays): x = l(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x=self.dropout(x)
        x=self.pred(x)
        x = self.pred2(x)
        return x


def lin(c_in, c_out, bn=True, dr=False):  # """Custom convolutional layer for simplicity."""
    layers=[]
    if dr > 0: layers.append(nn.Dropout(dr))
    layers.append(nn.Linear(c_in, c_out, bias=not bn))
    if bn: layers.append(nn.BatchNorm1d(c_out))
    layers.append(nn.ReLU())
    return layers

class linNet(nn.Module):
    def __init__(self, cfg):
        super(linNet, self).__init__()
        n_out = cfg["ds"][1]
        lw = cfg["aez"] * np.array([1] + cfg["llay"] + [1])
        lw[-1] = n_out
        lins = []
        for j in range(len(lw) - 1):
            dobn = cfg["lben"] and not (cfg["dben"] == 2 and (j + 1 == len(lw) - 1))
            lins += lin(lw[j], lw[j + 1], dobn, cfg["ldrop"][j])
        self.lays = nn.Sequential(*lins)

    def forward(self, x):
        x = self.lays(x)
        return x