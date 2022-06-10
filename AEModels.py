#Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
#Licence: Use it however you like, but cite the paper :-)

#Autoencoder models and training

import numpy as np
import pickle,os,copy
import torch.optim as optim
import torch.cuda.amp as tca
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn.functional as F
from aecyc.OnlineRep import imgutil


class AEDisc(nn.Module):
    def __init__(self, cfg, input_size=(1, 32, 32)):
            super(AEDisc, self).__init__()
            output_size = 1
            self.input_size = input_size
            self.channel_mult = int(64 * cfg["netSi"])
            bn = lambda x: nn.BatchNorm2d(x) if cfg["aeganbn"] else nn.Identity()
            bn1d = lambda x: nn.BatchNorm1d(x) if cfg["aeganbn"] else nn.Identity()
            slope = 0.2
            self.conv = nn.Sequential(*[nn.Conv2d(in_channels=input_size[0], out_channels=self.channel_mult * 1, kernel_size=4, stride=2, padding=1), bn(self.channel_mult * 1), nn.LeakyReLU(slope, True),
                                        nn.Conv2d(self.channel_mult * 1, self.channel_mult * 2, 4, 2, 1), bn(self.channel_mult * 2), nn.LeakyReLU(slope, inplace=True),
                                        nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, 4, 2, 1), bn(self.channel_mult * 4), nn.LeakyReLU(slope, inplace=True),
                                        nn.Conv2d(self.channel_mult * 4, self.channel_mult * 8, 4, 2, 1), bn(self.channel_mult * 8), nn.LeakyReLU(slope, inplace=True),
                                        nn.Conv2d(self.channel_mult * 8, self.channel_mult * 8, 3, 2, 1), bn(self.channel_mult * 8), nn.LeakyReLU(slope, inplace=True)])
            # self.flat_fts = self.get_flat_fts(self.conv)
            self.nin = self.channel_mult * 8
            self.linear = nn.Sequential(nn.Linear(self.nin, output_size) )

    def forward(self, x):
        for il, l in enumerate(self.conv):
            x = l(x)
        x = torch.flatten(x, start_dim=1)  # x.view(-1, self.flat_fts)
        return self.linear(x)



class CNN_Encoder(nn.Module):
    def __init__(self, cfg, input_size=(1, 32, 32)):
        super(CNN_Encoder, self).__init__()
        output_size=cfg["aez"]
        self.input_size = input_size
        self.channel_mult = int(64*(cfg["netSi"]+0.25*int("F" in cfg["ds"][0])))
        bn = lambda x: nn.BatchNorm2d(x) if cfg["aebn"] else nn.Identity()
        bn1d = lambda x: nn.BatchNorm1d(x) if cfg["aebn"] else nn.Identity()
        slope=0.05
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels=input_size[0],out_channels=self.channel_mult*1,kernel_size=4,stride=2,padding=1), bn(self.channel_mult*1),nn.LeakyReLU(slope, True),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1), bn(self.channel_mult*2), nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1), bn(self.channel_mult*4), nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),bn(self.channel_mult*8), nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*8, 3, 2, 1), bn(self.channel_mult*8), nn.LeakyReLU(slope, inplace=True)])
        #self.flat_fts = self.get_flat_fts(self.conv)
        self.nin=self.channel_mult*8
        self.linear = nn.Sequential(nn.Linear(self.nin, output_size),bn1d(output_size),nn.LeakyReLU(slope),)


    def forward(self, x):
        for il,l in enumerate(self.conv):
            x = l(x)
        x = torch.flatten(x,start_dim=1)#x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_Decoder(nn.Module):
    def __init__(self, cfg):
        super(CNN_Decoder, self).__init__()
        self.input_dim = cfg["aez"] #cfg["aecfg"]["esize"]
        self.channel_mult = int(64*(cfg["netSi"]+0.25*int("F" in cfg["ds"][0])))
        self.fc_output_dim = self.channel_mult*16#self.input_dim#int(64*cfg["aecfg"]["netSi"]) #cfg["aecfg"]["esize"]#128#256
        self.fc = nn.Sequential(nn.Linear(self.input_dim, self.fc_output_dim),nn.BatchNorm1d(self.fc_output_dim),nn.ReLU(True))
        bn = lambda x:  nn.BatchNorm2d(x) if cfg["aebn"] else nn.Identity()
        slope=0.05
        self.deconv = nn.Sequential(*[nn.ConvTranspose2d(self.fc_output_dim,self.channel_mult * 8, 4, 2,1, bias=False), bn(self.channel_mult * 8), nn.LeakyReLU(slope),
            nn.ConvTranspose2d(self.channel_mult * 8,self.channel_mult * 4, 4, 2,1, bias = False), bn(self.channel_mult * 4), nn.LeakyReLU(slope),
            nn.ConvTranspose2d(self.channel_mult * 4,self.channel_mult * 2, 4, 2, 1, bias = False), bn(self.channel_mult * 2), nn.LeakyReLU(slope),
            nn.ConvTranspose2d(self.channel_mult * 2,self.channel_mult * 1, 4, 2, 1, bias = False), bn(self.channel_mult * 1), nn.LeakyReLU(slope),
            nn.ConvTranspose2d(self.channel_mult * 1,cfg["imCh"], 4, 2, 1, bias = False)])

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        for l in self.deconv: x=l(x)
        return F.tanh(x)

class AENetwork(nn.Module):
    def __init__(self, cfg):
        super(AENetwork, self).__init__()
        self.encoder = CNN_Encoder(cfg,input_size=(cfg["imCh"],32,32))
        self.decoder = CNN_Decoder(cfg)

    def enc(self, x): return self.encoder(x)
    def dec(self, z): return self.decoder(z)
    def forward(self, x):
        z = self.enc(x)#.view(-1, 784))
        return self.dec(z),z

def getAEDat(netAE,dataset,encoded=True):
    netAE.eval()
    aencx=[]
    aency = []
    def nor(x): return (x - np.mean(x, axis=(0, 2, 3), keepdims=True)) / (np.std(x, axis=(0, 2, 3), keepdims=True) + 1e-7)
    with tca.autocast():
        with torch.no_grad():
          for i, data in enumerate(dataset):
             x = data[0].cuda()
             selfx,code = netAE(x)
             tosave=code if encoded else selfx
             aencx.append(tosave.detach().cpu().numpy())
             aency.append(np.copy(data[1].cpu().numpy()))
    return np.concatenate(aencx,axis=0),np.concatenate(aency,axis=0)

def getAEDatIter(netAE,trdata,tedata,encoded=True,cds=None):
    aetrX, aetrY=getAEDat(netAE, trdata, encoded=encoded)
    aetr_iter = cds(aetrX, aetrY, True, False)
    aeteX, aeteY = getAEDat(netAE, tedata, encoded=encoded)
    aete_iter = cds(aeteX, aeteY, False, False)
    return aetr_iter,aete_iter


def runAE(cfg,dataset,tedataset,sname,cget,picname):
    getM =getAEModel
    netAE, acfg = getM(cfg, dataset, sname, cget,picname)
    trds, teds = getAEDat(netAE, dataset), getAEDat(netAE, tedataset)
    #imgutil.makeAndStore(trds[:64], trds[:64], cfg["bFolder"] + "samples/", "AE" + picname + fs(cfg) + ".png")
    return trds,teds,acfg,netAE

def decay(ccf,epoch,optimizerCl):
    if ccf["opt"][0] == "S" and (epoch + 1) % (ccf["opt"][1] // 3+ccf["opt"][1]//10+2 ) == 0:
        for p in optimizerCl.param_groups: p['lr'] *= 0.1
        print("  D", np.round(optimizerCl.param_groups[0]['lr'],5))

def getAEModel(cfg, train_dataset, sname, cget,picname=""): #Co,val_datasetMa,resFolder
    ccf=cfg["aecfg"]
    netAE = AENetwork(cfg).cuda()
    if cfg["aeGAN"][0]:
        netD=AEDisc(cfg).cuda()
        optimizerD = optim.Adam(netD.parameters(), lr=cfg["aeGAN"][1]["lr"], betas=(0.5, 0.999))
        optimizerG = optim.Adam(netAE.parameters(), lr=cfg["aeGAN"][1]["lr"], betas=(0.5, 0.999))
        criterion = nn.BCEWithLogitsLoss()
        real_label, fake_label = 1., 0.
        gloss,drloss,dfloss=0,0,0

    if ccf["opt"][0] == "S": optimizerCl = optim.SGD(netAE.parameters(), lr=ccf["opt"][1], momentum=0.8, weight_decay=ccf["opt"][2])
    elif ccf["opt"][0] == "A": optimizerCl = optim.Adam(netAE.parameters(), ccf["opt"][1], weight_decay=ccf["opt"][2])
    else: "Error opt not found"
    closs, trep, loss = 0,  cfg["epA"], nn.MSELoss()#nn.CrossEntropyLoss()
    print("Train AE")
    scaler = tca.GradScaler()
    ulo = lambda x,t,e: 0.97*x+0.03*t.item() if e>1 else 0.85*x+0.15*t.item()
    torch.backends.cudnn.benchmark = True
    for epoch in range(trep):
            netAE.train()
            for i, data in enumerate(train_dataset):
               with tca.autocast():
                optimizerCl.zero_grad()
                x=data[0].cuda()
                outAE,lo=netAE(x)
                errD_real = loss(torch.flatten(outAE,1),torch.flatten(x,1))
                scaler.scale(errD_real).backward()
                scaler.step(optimizerCl)
                scaler.update()
                if cfg["aeGAN"][0]:
                    ## Train with all-real batch
                    netD.zero_grad()
                    b_size = x.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float).cuda()
                    outreal = netD(x).view(-1)
                    errD_real = criterion(outreal, label)
                    scaler.scale(errD_real).backward()

                    ## Train with all-fake batch
                    label.fill_(fake_label)
                    outfake = netD(outAE.detach()).view(-1)
                    errD_fake = criterion(outfake, label)
                    scaler.scale(errD_fake).backward()
                    scaler.step(optimizerD)

                    # (2) Update G network: maximize log(D(G(z)))
                    optimizerG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    outAE, _ = netAE(x)
                    outfake = netD(outAE).view(-1)
                    errG = criterion(outfake, label)
                    scaler.scale(errG).backward()
                    scaler.step(optimizerG)
                    scaler.update()

                    gloss = ulo(gloss, errG, epoch)
                    drloss = ulo(drloss, errD_real, epoch)
                    dfloss = ulo(dfloss, errD_fake, epoch)

                closs = ulo(closs,errD_real,epoch)
            decay(ccf,epoch,optimizerCl)
            netAE.eval()
            if (epoch % 2 == 0 and epoch<=10) or (epoch % 10==0 and epoch>10):
                print(epoch, "AE", np.round(np.array([closs]+([gloss,drloss,dfloss] if cfg["aeGAN"][0] else [])), 5), cfg["pr"])
                if np.isnan(closs):
                    print("Failed!!!")
                    return None,None
    lcfg = {"AELo": closs}
    if cfg["aeGAN"][0]: lcfg={**lcfg,**{"glo":gloss,"drlo":drloss,"dflo":dfloss}}
    netAE.eval()
    return netAE, lcfg


