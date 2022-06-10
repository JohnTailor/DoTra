#Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
#Licence: Use it however you like, but cite the paper :-)

#Training of classifiers (and also DoTra on paired samples)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as tca
from classifierModels import worNet,linNet
from torch.utils.data import Dataset,TensorDataset
from doTraModel import Trans
import imgutil

niter = 1e10

def decay(ccf,epoch,optimizerCl,warmup,trep):
    if epoch==warmup[0]:
        for p in optimizerCl.param_groups: p['lr'] *= (warmup[1] if ccf["opt"][0] == "S" else warmup[1]/3)
        print("  W", np.round(optimizerCl.param_groups[0]['lr'],5))
    if ccf["opt"][0] == "S" and (epoch + 1) % int(trep// 3+10+warmup[0] ) == 0:
        for p in optimizerCl.param_groups: p['lr'] *= 0.1
        print("  D", np.round(optimizerCl.param_groups[0]['lr'],5))

def getSingleAcc(net, dsx, labels, pool=None):
  with tca.autocast():
    outputs = net(dsx)
    _, predicted = torch.max(outputs.data, 1)
    correct = torch.eq(predicted,labels).sum().item()
    return correct


def getAcc(net, dataset,  niter=10000,cfg=None):
    correct,total = 0,0
    net.eval()
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                dsx,dsy = data[0].cuda(),data[1].cuda().unsqueeze(-1)
                outputs = net(dsx)  # if useAtt:                #     errD_real = loss(output[0], dsy.long())+loss(output[1], dsy.long())                #     output=output[1] #prediction outputs                # else:
                total += dsy.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += torch.eq(predicted, dsy.squeeze().long()).sum()
                if cit>=niter: break
    return float((correct*1.0/total).cpu().numpy())

def getCorr(net, dataset):
    correct = []
    conf=[]
    net.eval()
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                dsx,dsy = data[0].cuda(),data[1].cuda().unsqueeze(-1)
                outputs = net(dsx)
                preconf, predicted = torch.max(outputs, 1)
                correct.append(torch.eq(predicted, dsy.squeeze().long()).detach().cpu().numpy())
                conf.append(preconf.detach().cpu().numpy())
    return np.concatenate(correct,axis=0),np.concatenate(conf,axis=0)


def getCls(net, dataset):
    net.eval()
    bx,by=[],[]
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                dsx,dsy = data[0].cuda(),data[1].cuda().unsqueeze(-1)
                outputs = net(dsx)  # if useAtt:                #     errD_real = loss(output[0], dsy.long())+loss(output[1], dsy.long())                #     output=output[1] #prediction outputs                # else:
                _, predicted = torch.max(outputs, 1)
                by.append(predicted.detach().cpu().numpy())
                bx.append(data[0].cpu().numpy())
    return np.concatenate(bx,axis=0),np.concatenate(by,axis=0)

def setEval(netCl):
        netCl.eval()
        for name, module in netCl.named_modules():
            if isinstance(module, nn.Dropout): module.p = 0
            elif isinstance(module, nn.LSTM): module.dropout = 0 #print("zero lstm drop") #print("zero drop")
            elif isinstance(module, nn.GRU): module.dropout = 0


def getTrans(cfg,train_dataset,val_dataset,dat,traname,cget,selfTra=False):
    ccf=cfg["trcfg"]
    NETWORK = Trans #if "V" in ccf["netT"] else (res.ResNet10 if ccf["netT"] == "R10" else res.ResNet18)
    netCl = NETWORK(cfg).cuda()

    closs, teaccs, trep,  clr,telo = 0, [], cfg["epC"], ccf["opt"][1],0
    loss = nn.MSELoss() if cfg["traLo"] else  nn.L1Loss()
    warmup = (max(2,trep//40), 10)
    #if ccf["opt"][0] == "S": optimizerCl = optim.SGD(netCl.parameters(), lr=ccf["opt"][1]/warmup[1], momentum=0.8, weight_decay=ccf["opt"][2])
    #elif ccf["opt"][0] == "A": #else: "Error opt not found"
    optimizerCl = optim.Adam(netCl.parameters(), ccf["opt"][1], weight_decay=ccf["opt"][2])
    print("Train Trans",ccf)
    scaler = tca.GradScaler()
    nDom=2#len(cfg["trans"])+1-cfg["nFor"] #Last for testing
    inds=np.zeros(3,dtype=np.int)


    torch.backends.cudnn.benchmark = True
    for epoch in range(trep):
        netCl.train()
        for i, data in enumerate(train_dataset):
          with tca.autocast():
            optimizerCl.zero_grad()
            xdata=data[:-1]
            if cfg["singleIn"]:
                inds[0] = np.random.choice(nDom - 1)  # -4 = X0, -3=X1, -2=XTe, -1=XPreTest
                inds[1] = inds[0]
                inds[2] = inds[1] + 1
            else:
                inds[0]=np.random.choice(nDom - 2) #-4 = X0, -3=X1, -2=XTe, -1=XPreTest
                inds[1] = inds[0] + 1
                if cfg["ranCh"]: inds[1] +=np.random.choice(nDom - 2-inds[0])
                inds[2]=inds[1]+1
            dsx=[xdata[cind].cuda() for cind in inds] #dsy = data[1].cuda()
            output = netCl(dsx[:2])
            errD_real = loss(output,dsx[-1]) # errD_real.backward()            # optimizerCl.step()
            scaler.scale(errD_real).backward()
            scaler.step(optimizerCl)
            scaler.update()
            closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()

        decay(ccf,epoch,optimizerCl,warmup,trep)
        netCl.eval()
        #if epoch%16==0: store(pre="Ep_" + str(epoch)+"_",dirapp="Tmp",output=output,xdata=xdata)
        if (epoch % 2 == 0 and epoch<=10) or (epoch % 10==0 and epoch>10):
            print(epoch, np.round(np.array([closs]),5), cfg["pr"])#teAccs[-1], clAcc(train_dataset)
            if np.isnan(closs):
                print("Failed!!!")
                return None,None,None
    def getLo(ds,off=0):
        telo,nele=0,0
        for i, xdata in enumerate(ds):
            with tca.autocast():
                with torch.no_grad():
                    ainds = [nDom - 3+off, nDom - 2+off, nDom - 1+off]  # Shift by one to get test
                    dsx = [xdata[cind].cuda() for cind in ainds]
                    output = netCl(dsx[:2])
                    telo += loss(output, dsx[-1])
                    nele+=dsx[0].shape[0]
        return (telo/nele).item()

    def transform(cfg, orgTransModel, traX, traY):
        def cds(X, Y, shuffle=False, norm=True):
            noX = imgutil.nor(X.astype(np.float32)) if norm else X
            ds = TensorDataset(torch.from_numpy(noX), torch.from_numpy(Y))
            return torch.utils.data.DataLoader(ds, batch_size=cfg["batchSize"], shuffle=shuffle, num_workers=0)

        cajx, cajy = [], []
        orgTransModel.eval()
        for data in cds(traX, traY, shuffle=False):
            with tca.autocast():
                # with torch.no_grad():
                dsx = data[0].cuda()
                out1 = orgTransModel([None, dsx])
                output = orgTransModel([None, out1]).detach().cpu()
                cajx.append(output.clone().numpy())
                cajy.append(data[-1].clone().numpy())
        return cds(np.concatenate(cajx, axis=0), np.concatenate(cajy, axis=0))

    lcfg = { "trLo": closs,"tetrLo": getLo(train_dataset),"teteLo": getLo(val_dataset)}#,"D1AccTra":traAcc
    setEval(netCl)
    return netCl, lcfg,False

def getclassifier(cfg,train_dataset,val_dataset,sname,getc,save=False,loadCl=True,useLat=False):
    print(sname,"Cl")
    ccf=cfg["clcfg"]
    if useLat: NETWORK = linNet
    else: NETWORK = worNet #if "V" in ccf["netT"]  else (res.ResNet10 if ccf["netT"] == "R10" else res.ResNet18)
    netCl = NETWORK(cfg).cuda()

    closs, teaccs, trep, loss, clr = 0, [], cfg["epC"], nn.CrossEntropyLoss(), ccf["opt"][1]
    warmup = (max(2,trep//40), 10)
    if ccf["opt"][0] == "S": optimizerCl = optim.SGD(netCl.parameters(), lr=ccf["opt"][1]/warmup[1], momentum=0.9, weight_decay=ccf["opt"][2]) #elif ccf["opt"][0] == "A": optimizerCl = optim.Adam(netCl.parameters(), ccf["opt"][2], weight_decay=ccf["opt"][3])
    else: "Error opt not found"
    print("Train CL",sname,ccf)
    scaler = tca.GradScaler()
    teAccs=[]
    clAcc = lambda dataset: getAcc(netCl, dataset,  niter=niter,cfg=cfg)
    crolo=nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True
    for epoch in range(trep):
        netCl.train()
        for i, data in enumerate(train_dataset):
          with tca.autocast():
            optimizerCl.zero_grad()
            dsx,dsy = data[0].cuda(),data[1].cuda()
            output = netCl(dsx)
            errD_real = crolo(output,dsy.long())
            scaler.scale(errD_real).backward()
            scaler.step(optimizerCl)
            scaler.update()
            closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()
        decay(ccf,epoch,optimizerCl,warmup,trep)
        netCl.eval()
        teAccs.append(clAcc(val_dataset))
        if (epoch % 2 == 0 and epoch<=10) or (epoch % 10==0 and epoch>10):
            print(epoch, np.round(np.array([closs, teAccs[-1], clAcc(train_dataset)]), 5), cfg["pr"])
            if np.isnan(closs):
                print("Failed!!!")
                return None,None,None
    mteA=np.max(np.array(teAccs))
    lcfg = {"teA": teAccs[-1], "trA": clAcc(train_dataset), "Lo": closs,"mteA":mteA}
    setEval(netCl)
    return netCl, lcfg


def getLinCl(cfg,train_dataset,val_dataset,sname,getc,save=True,loadCl=True):
    ccf=cfg["clcfg"]
    from aecyc.latAEModels import LinCl
    netCl = LinCl(cfg).cuda()
    closs, teaccs, trep, loss, clr = 0, [], cfg["epC"], nn.CrossEntropyLoss(), ccf["opt"][1]/4 #Train just 1/2 as long
    warmup = (max(2,trep//40), 10)
    if ccf["opt"][0] == "S": optimizerCl = optim.SGD(netCl.parameters(), lr=clr/warmup[1], momentum=0.8, weight_decay=ccf["opt"][2]/5) #elif ccf["opt"][0] == "A": optimizerCl = optim.Adam(netCl.parameters(), ccf["opt"][2], weight_decay=ccf["opt"][3])
    else: "Error opt not found"
    print("Train CL",sname,ccf)
    scaler = tca.GradScaler()
    teAccs=[]
    clAcc = lambda dataset: getAcc(netCl, dataset,  niter=niter,cfg=cfg)
    crolo=nn.CrossEntropyLoss()
    for epoch in range(trep):
        netCl.train()
        for i, data in enumerate(train_dataset):
          with tca.autocast():
            optimizerCl.zero_grad()
            dsx,dsy = data[0].cuda(),data[1].cuda()
            output = netCl(dsx)
            errD_real = crolo(output,dsy.long())
            scaler.scale(errD_real).backward()
            scaler.step(optimizerCl)
            scaler.update()
            closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()
        decay(ccf,epoch,optimizerCl,warmup,trep)
        netCl.eval()
        teAccs.append(clAcc(val_dataset))
        if (epoch % 2 == 0 and epoch<=10) or (epoch % 10==0 and epoch>10):
            print(epoch, np.round(np.array([closs, teAccs[-1], clAcc(train_dataset)]), 5), cfg["pr"])
            if np.isnan(closs):
                print("Failed!!!")
                return None,None,None
    lcfg = {"LiteA": clAcc(val_dataset), "LitrA": clAcc(train_dataset), "LiLo": closs}
    setEval(netCl)
    return netCl, lcfg