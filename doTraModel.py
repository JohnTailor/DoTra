#Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
#Licence: Use it however you like, but cite the paper :-)

import torch.nn as nn
import torch.nn.functional as F
import torch

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True): #"""Custom deconvolutional layer for simplicity."""
    layers = [nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not bn)]
    if bn: layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):#"""Custom convolutional layer for simplicity."""
    layers = [nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not bn)]
    if bn: layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Trans(nn.Module):

    def __init__(self, cfg):
        super(Trans, self).__init__()
        self.cfg=cfg
        conv_dim = int(cfg["convdim"]*cfg["netSi"])
        self.leak=cfg["tleak"]
        # encoding blocks
        self.in1=cfg["singleIn"]
        self.sym=cfg["sym"]
        insym=cfg["imCh"]*(1+2*int(cfg["sym"]==1) +int(cfg["sym"]==2)+ 3*int(cfg["sym"]==3))
        if self.in1:
            self.conv1 = conv(insym, conv_dim, 4)
            self.conv2 = conv(conv_dim , conv_dim * 2 , 4)
        else:
            self.conca = cfg["conca"]
            co = self.conca
            self.conv1 = conv(insym*(co+1), conv_dim//(2-co), 4)
            self.conv2 = conv(conv_dim//(2-co), conv_dim*2//(2-co), 4)

        # residual blocks
        if cfg["resB"]:
            self.conv3= BasicBlock(conv_dim*2, conv_dim*2)
            self.conv3a = BasicBlock(conv_dim * 2, conv_dim * 2, 3) if cfg["nExLay"] else nn.Identity()
            self.conv4 = BasicBlock(conv_dim * 2, conv_dim * 2)
        else:
            self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
            self.conv3a = conv(conv_dim*2, conv_dim*2, 3, 1, 1) if cfg["nExLay"] else nn.Identity()
            self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        
        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, cfg["imCh"], 4, bn=False)

    def geto(self,inx):
        out = F.leaky_relu(self.conv1(inx), self.leak)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), self.leak)  # (?, 128, 8, 8)
        return out

    def forward(self, x):
        x1,x2=x
        if self.in1:
            if self.sym:
                xsym = torch.flip(x2, dims=(-1,))
                if self.sym==1: x2=torch.cat([x2,xsym,torch.flip(x2, dims=(-2,))],dim=1)
                if self.sym == 2: x2 = torch.cat([x2,torch.flip(xsym,dims=(-2,))], dim=1)
                if self.sym==3: x2 = torch.cat([x2, xsym, torch.flip(x2, dims=(-2,)), torch.flip(xsym, dims=(-2,))], dim=1)
            out = self.geto(x2)
        else:
            if self.sym>0: print("must flip etc for each input - not implemented see above how to do it")
            if self.conca:
                x=torch.cat([x1,x2],dim=1)
                out=self.geto(x)
            else:
                out = torch.cat([self.geto(x1), self.geto(x2)], dim=1)
        
        out = F.leaky_relu(self.conv3(out), self.leak)    # ( " )
        out = F.leaky_relu(self.conv3a(out), self.leak) if self.cfg["nExLay"] else out
        out = F.leaky_relu(self.conv4(out), self.leak)    # ( " )
        
        out = F.leaky_relu(self.deconv1(out), self.leak)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 3, 32, 32)
        return out



class D1(nn.Module):
    """Discriminator for mnist."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out

class D2(nn.Module):
    """Discriminator for svhn."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out