#Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
#Licence: Use it however you like, but cite the paper :-)

#Based on https://github.com/yunjey/mnist-svhn-transfer/


import torch
import torch.nn as nn
import os
import pickle
import numpy as np

from torch.autograd import Variable
from torch import optim
import torch.cuda.amp as tca

from trainClassifiers import getAcc
from latAEModels import G, D
from trainClassifiers import decay
from classifierModels import worNet

def getTrAcc( cfg, trds,val_dataset):
    netCl = worNet(cfg).cuda()
    ccf = cfg["clcfg"]
    closs, teaccs, trep, loss, clr = 0, [], cfg["epC"]//3, nn.CrossEntropyLoss(), ccf["opt"][1]
    # optimizerCl = optim.SGD(netCl.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-5)  # elif ccf["opt"][0] == "A": optimizerCl = optim.Adam(netCl.parameters(), ccf["opt"][2], weight_decay=ccf["opt"][3])
    warmup = (max(2, trep // 40), 10)
    optimizerCl = optim.SGD(netCl.parameters(), lr=ccf["opt"][1] / warmup[1], momentum=0.9, weight_decay=ccf["opt"][2])
    scaler = tca.GradScaler()
    clAcc = lambda dataset: getAcc(netCl, dataset, niter=9999, cfg=cfg)
    crolo = nn.CrossEntropyLoss()
    closs = 0
    for epoch in range(trep):
        netCl.train()
        for i, data in enumerate(trds):
            with tca.autocast():
                optimizerCl.zero_grad(set_to_none=True)
                dsx, dsy = data[0].cuda(), data[1].cuda()
                output = netCl(dsx)
                errD_real = crolo(output, dsy.long())
                scaler.scale(errD_real).backward()
                scaler.step(optimizerCl)
                scaler.update()
                closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()
        decay(ccf, epoch, optimizerCl, warmup, trep)
        netCl.eval()
        if epoch<2 or epoch==trep-1 or epoch%15==0:
            print("Train Test CL","ep", epoch, closs, clAcc(val_dataset))
    return clAcc(val_dataset),netCl


class Solver(object):
    def __init__(self, cfg):
        self.cfg=cfg
        self.g12,self.g21 = None,None
        self.d1,self.d2 = None,None
        self.g_optimizer,self.d_optimizer = None,None
        self.num_classes = cfg["ds"][1]
        self.batch_size = cfg["batchSize"]



    def build_model(self): # """Builds a generator and a discriminator."""
        self.g12 = G(self.cfg).cuda()
        self.g21 = G(self.cfg).cuda()
        if self.cfg["d1"]: self.d1 = D(self.cfg, use_labels=self.cfg["useLab"] in [1,2]).cuda()
        self.d2 = D(self.cfg, use_labels=self.cfg["useLab"]==2).cuda()
        if self.cfg["useLab"] == 3: self.d1cl = D(self.cfg, use_labels=1).cuda()
        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params =  list(self.d2.parameters())
        if self.cfg["d1"]: d_params+=  list(self.d1.parameters() )
        if self.cfg["useLab"] == 3: d_params+=  list(self.d1cl.parameters() )
        self.d_optimizer = optim.Adam(d_params, self.cfg["DGlr"][0], self.cfg["DGBeta12"][0])
        self.g_optimizer = optim.Adam(g_params, self.cfg["DGlr"][1],self.cfg["DGBeta12"][1])


    def to_var(self, x): return Variable(x.cuda()) # """Converts numpy to variable."""
    #def to_data(self, x): return x.cpu().data.numpy() #"""Converts variable to numpy."""
    def reset_grad(self):
        self.g_optimizer.zero_grad(set_to_none=True)
        self.d_optimizer.zero_grad(set_to_none=True)
        #if self.cfg["useLab"] == 3: self.d3_optimizer.zero_grad()

    def decay(self,epoch,total):
        if epoch>total*self.cfg["lrdecay"]:
            for opt in [self.g_optimizer,self.d_optimizer]:
                for g in opt.param_groups:
                    g['lr']=0.85*g['lr']


    def train(self,netAEorg,netAEdom,aetr_iter,aete_iter,aemodtr_iter,modteX,modteY,cget,nd):        #drift_iter = iter(self.adaDom_loader)        #orgDom_iter = iter(self.orgDom_loader)
        othlo, milo,bacc = -1, 1e99,0
        reclo, reclo2 = torch.zeros(1), torch.zeros(1)

        print("Train Cyc GAN")
        if self.cfg["useClLo"]:
            from trainClassifiers import getLinCl
            clloss = nn.CrossEntropyLoss()
            if self.cfg["trainCl"]:
                ccf = self.cfg["clcfg"]
                from latAEModels import LinCl

                clDom = LinCl(self.cfg).cuda()
                loOrg, loDom= nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
                clr =  ccf["opt"][1] / 100
                if self.cfg["trainCl"]!=3:
                    clOrg = LinCl(self.cfg).cuda()
                    optOrg = optim.SGD(clOrg.parameters(), lr=clr, momentum=0.8, weight_decay=ccf["opt"][2] / 5) #clr / warmup[
                else:
                    linCl, lincfg = getLinCl(self.cfg, aetr_iter, aete_iter,"None", cget, save=True, loadCl=True)
                optDom = optim.SGD(clDom.parameters(), lr=clr, momentum=0.8, weight_decay=ccf["opt"][2] / 5)
            else:
                 linCl, lincfg=getLinCl(self.cfg, aetr_iter, aete_iter, "None", cget, save=True, loadCl=True)
        self.build_model()
        useLabLoss=nn.CrossEntropyLoss() # loss if use_labels = True
        ax, ay = [], []
        tries = 0
        niterEp = nd // self.cfg["batchSize"]
        train_iters = self.cfg["epG"] * niterEp + self.cfg["ntries"][0] * self.cfg["ntries"][1]  # config.train_iters
        labSmo = lambda sh: 2*(torch.rand(sh.shape[0]).cuda()-0.5)*self.cfg["labSmo"] if self.cfg["labSmo"]>0 else 0
        recF= torch.square if self.cfg["recLo"] %10== 2 else torch.abs
        recLoFct = lambda x: recF(torch.mean(torch.abs(x),dim=1)) if self.cfg["recLo"]>=10 else recF(x)
        miter, siter = iter(aetr_iter), iter(aemodtr_iter)
        sca0,sca1,sca2,sca3=tca.GradScaler(),tca.GradScaler(),tca.GradScaler(),tca.GradScaler()
        def wrap(scaler,opt,lo):
            scaler.scale(lo).backward()
            scaler.step(opt)
            scaler.update()
        for step in range(train_iters + 1):  # # reset data_iter for each epoch
           with tca.autocast():
            try:
                adaDom, s_labels = next(siter)  ## load adaDom and orgDom dataset
                orgDom, m_labels = next(miter)
            except StopIteration:
                miter, siter = iter(aetr_iter), iter(aemodtr_iter)
                adaDom, s_labels = next(siter)
                orgDom, m_labels = next(miter)
                self.decay(step//niterEp, self.cfg["epG"])

            orgDom,adaDom = orgDom.float(),adaDom.float()
            if step == 0: code_org, code_dom = orgDom.clone().cuda(), adaDom.clone().cuda() #save for outputting images
            adaDom, s_labels = self.to_var(adaDom), self.to_var(s_labels).long().squeeze()
            orgDom, m_labels = self.to_var(orgDom), self.to_var(m_labels.long())

            if self.cfg["useLab"]: orgDom_fake_labels = self.to_var(torch.Tensor([self.num_classes] * adaDom.size(0)).long())
            if self.cfg["useLab"] == 2: adaDom_fake_labels = self.to_var(torch.Tensor([self.num_classes] * orgDom.size(0)).long())

            # ============ train D ============#
            # train with real images
            self.reset_grad()
            d1_loss = 0
            if self.cfg["d1"]:
                out = self.d1(orgDom)
                if self.cfg["useLab"] ==1 or self.cfg["useLab"] ==2: d1_loss = useLabLoss(out, m_labels)
                else: d1_loss = torch.mean((out - 1+labSmo(out)) ** 2)
            out = self.d2(adaDom)
            if self.cfg["useLab"] == 2: d2_loss = useLabLoss(out, s_labels)
            else: d2_loss = torch.mean((out - 1+labSmo(out)) ** 2)
            d_orgDom_loss, d_adaDom_loss, d_real_loss = d1_loss, d2_loss, d1_loss + d2_loss
            if self.cfg["useLab"] == 3:
                out = self.d1cl(orgDom)
                d_real_loss += useLabLoss(out, m_labels)
            wrap(sca0,self.d_optimizer,d_real_loss)


            # train with fake images
            self.reset_grad()
            fake_adaDom = self.g12(orgDom)
            out = self.d2(fake_adaDom)
            if self.cfg["useLab"] == 2: d2_loss = useLabLoss(out, adaDom_fake_labels)
            else: d2_loss = torch.mean((out+labSmo(out)) ** 2)
            fake_orgDom = self.g21(adaDom)
            if self.cfg["d1"]:
                out = self.d1(fake_orgDom)
                if self.cfg["useLab"] ==1 or self.cfg["useLab"] ==2: d1_loss = useLabLoss(out, orgDom_fake_labels)
                else: d1_loss = torch.mean((out+labSmo(out)) ** 2)
            else: d1_loss=0
            d_fake_loss = d1_loss + d2_loss
            if self.cfg["useLab"] == 3:
                out = self.d1cl(fake_orgDom)
                d_fake_loss += useLabLoss(out, orgDom_fake_labels)
            # d_fake_loss.backward()
            # self.d_optimizer.step()
            wrap(sca1, self.d_optimizer, d_fake_loss)

            # ============ train G ============#
            # train orgDom-adaDom-orgDom cycle
            self.reset_grad()
            fake_adaDom = self.g12(orgDom)
            out = self.d2(fake_adaDom)
            if self.cfg["useLab"] == 2: g_loss = useLabLoss(out, m_labels)
            else: g_loss = torch.mean((out - 1) ** 2)
            if self.cfg["useRec"] > 0:
                reconst_orgDom = self.g21(fake_adaDom)
                reclo= self.cfg["cycFac"][0]*self.cfg["useRec"] * torch.mean(recLoFct(orgDom - reconst_orgDom))
                g_loss += reclo
            if self.cfg["useLab"] == 3:
                out = self.d1cl(reconst_orgDom)
                g_loss += useLabLoss(out, m_labels)
            if self.cfg["useClLo"]:
                if self.cfg["trainCl"]:
                    actOrg = clOrg(reconst_orgDom)  # subtract loss on original maybe
                    actDom = clDom(fake_adaDom)  # subtract loss on original maybe
                    if self.cfg["smoo"]:
                        actOrg=actOrg+torch.mean(torch.abs(actOrg.detach()),dim=0)*self.cfg["smoo"]
                        actDom = actDom + torch.mean(torch.abs(actDom.detach()),dim=0) * self.cfg["smoo"]
                    li_loss = self.cfg["trainCl"] * (clloss(actOrg, m_labels) + clloss(actDom, m_labels))
                else:
                    acti = linCl(reconst_orgDom)  # subtract loss on original maybe
                    if self.cfg["smoo"]:
                        acti=acti+torch.mean(torch.abs(acti.detach()),dim=0)*self.cfg["smoo"]
                    li_loss=self.cfg["useClLo"]*clloss(acti, m_labels)
                g_loss += li_loss

            wrap(sca2, self.g_optimizer, g_loss)


            # train adaDom-orgDom-adaDom cycle
            self.reset_grad()
            fake_orgDom = self.g21(adaDom)
            if self.cfg["d1"]:
                out = self.d1(fake_orgDom)
                if self.cfg["useLab"] == 2: g_loss = useLabLoss(out, s_labels)
                else: g_loss = torch.mean((out - 1) ** 2)
            else: g_loss=0
            if self.cfg["useRec"] > 0:
                reconst_adaDom = self.g12(fake_orgDom)
                reclo2=self.cfg["cycFac"][1] * self.cfg["useRec"] * torch.mean(recLoFct(adaDom - reconst_adaDom))
                g_loss += reclo2

            wrap(sca3, self.g_optimizer, g_loss)

            if self.cfg["useClLo"]:
                if self.cfg["trainCl"]:
                    def trCl(cl,dat, lo, opt):
                        opt.zero_grad(set_to_none=True)
                        cl.train()
                        act=cl(dat)
                        clo = lo(act, m_labels)
                        clo.backward()
                        opt.step()
                        cl.eval()
                    if not self.cfg["trainOrg"]==2: trCl(clOrg, reconst_orgDom.detach(), loOrg, optOrg)
                    if self.cfg["trainOrg"]==1:    trCl(clOrg, orgDom.detach(), loOrg, optOrg)
                    trCl(clDom, fake_adaDom.detach(), loDom, optDom)


            if (step + 1) % self.cfg["ntries"][0] == 0:  # print the log info self.log_step
                useLat = self.cfg["useLat"]

                def getaeds(ds):
                    ax0, ax1, ay = [], [], []
                    self.g12.eval()
                    for bx, by in ds:
                       with tca.autocast():
                        cx = bx.float().cuda()
                        fake_code_dom = self.g12(cx)
                        if not useLat:
                            orgX = netAEorg.dec(cx).detach().cpu().numpy()
                            domGenX = netAEdom.dec(fake_code_dom).detach().cpu().numpy()
                        ax0.append(orgX if not useLat else bx)
                        ax1.append(domGenX if not useLat else fake_code_dom.detach().cpu().numpy())
                        ay.append(by.detach().cpu().numpy())
                    self.g12.train()
                    return [np.concatenate(cx, axis=0) for cx in [ax0, ax1, ay]]

                if (step + 1) % (10*self.cfg["ntries"][0]) == 0:
                    print('Step [%d/%d], Losses: d_real: %.4f, d_OrgDom: %.4f, d_AdaDom: %.4f, '
                          'd_fake: %.4f, g: %.4f, r: %.4f, r2: %.4f' % (step + 1, train_iters, d_real_loss.item(), d_orgDom_loss.item() if self.cfg["d1"] else -1, d_adaDom_loss.item(), d_fake_loss.item(), g_loss.item(), reclo.item(), reclo2.item()),self.cfg["pr"])
                    if self.cfg["useClLo"]: print("LinCl Loss",li_loss.item())

                clo = g_loss.item()


                if (step // niterEp >= self.cfg["epG"] and milo * 0.85 > clo):
                    milo = clo
                    tries += 1
                    [ax0, ax1, ay] = getaeds(aetr_iter)
                    [atex0, atex1, atey] = getaeds(aete_iter)
                    if tries == self.cfg["ntries"][1]: break


        othlo = {"DOLo": d_orgDom_loss.item() if self.cfg["d1"] else 0, "DDLo": d_adaDom_loss.item(), "DFLo": d_fake_loss.item()}  # "DRLo":d_real_loss.item(),
        return othlo,ax0, ax1, ay, atex0, atex1, atey



