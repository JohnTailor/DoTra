# Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
# Licence: Use it however you like, but cite the paper :-)

#Main routine to train models

import sklearn

import torch
from torch.utils.data import Dataset, TensorDataset
import torch.cuda.amp as tca

from optCycEncoded import Solver
import dutils
import AEModels as aut
import trainClassifiers,imgutil
from imgutil import *


def trainOne(cfg):
    def cds(X, Y, shuffle=False, norm=True):
        noX = imgutil.nor(X.astype(np.float32)) if norm else X
        ds = TensorDataset(torch.from_numpy(noX), torch.from_numpy(Y))
        return torch.utils.data.DataLoader(ds, batch_size=cfg["batchSize"], shuffle=shuffle, num_workers=0)

    def cds2(X, Y, shuffle=False):
        noX = [imgutil.nor(cX.astype(np.float32)) for cX in X]
        ds = TensorDataset(*[torch.from_numpy(cX) for cX in noX], torch.from_numpy(Y))
        return torch.utils.data.DataLoader(ds, batch_size=cfg["batchSize"], shuffle=shuffle, num_workers=0)

    def modXY(toModteX,modteY):
        t = cfg["trans"][0]
        modteX=applyOp(toModteX,t[0],t[1])
        return modteX,modteY

    def cdsMod(X, Y):
        X = imgutil.nor(X)
        allX, allY = [X], [Y]
        for t in cfg["transEv"]:
            modtrX, Y = modXY(allX[-1].astype(np.float32), allY[-1])
            allX.append(modtrX)
            allY.append(Y)
        return allX, allY

    def getAE(cfg, trX, trY, teX, teY, sname, dotrain,picname=""):
        origtr_iter = cds(trX, trY, True)
        origte_iter = cds(teX, teY, False)
        (aetrX, aetrY), (aeteX, aeteY), acfg, netAEorg = aut.runAE(cfg, origtr_iter, origte_iter,  sname, dotrain,picname)
        aetr_iter = cds(aetrX, aetrY, True, False)
        aete_iter = cds(aeteX, aeteY, False, False)
        return aetr_iter, aete_iter, netAEorg



    cget=True
    #Get unmodified, raw training data and a base Classifier
    totdat= int(cfg["ds"][2])
    (trX, trY), (teX, teY) = dutils.getFullDS(cfg, totdat, None, cget)

    if cfg["baseZoom"]!=1:
        print("Zooming data for Proposition test")
        trX=applyOp(trX, "zoo1", cfg["baseZoom"])
        teX = applyOp(teX, "zoo1", cfg["baseZoom"])

    disoff = int(totdat * cfg["distinctDat"]*0.5)
    nd = totdat-disoff
    baseCl, baseClRes = trainClassifiers.getclassifier(cfg, cds(trX, trY, True), cds(teX, teY, False), None, getc=False, loadCl=False, save=False, useLat=False) #get base classifier (only used for reference)

    toModtrX, modtrY = np.copy(trX[disoff:nd + disoff]), np.copy(trY[disoff:nd + disoff])
    toModteX, modteY = np.copy(teX), np.copy(teY)
    trX, trY = trX[:nd], trY[:nd]

    #Get auto encoding of raw training data and domain adapted data
    aetr_iter, aete_iter, netAEorg = getAE(cfg,trX, trY, teX, teY, sname=None, dotrain=cget,picname="onRawOrgDomain") #AE with tanh => -1,1
    t = cfg["trans"][0]

    modtrX, modtrY = modXY(toModtrX, modtrY)
    modteX, modteY = modXY(toModteX,modteY)

    failed = False
     # Full DoTra: autoencode, transform on latent, learn between domains
    aemodtr_iter, aemodte_iter, netAEdom = getAE(cfg,modtrX, modtrY, modteX, modteY, sname=None, dotrain=cget,picname="onRawAdaDomain")

    #Get Transformer between auto encodings of raw training data and for the same data in domain-adapted space
    solver = Solver(cfg) #if cfg["solvSim"]==0 else solverSimp.Solver(cfg)
    othlo, ax0, ax1, ay, atex0, atex1, atey = solver.train(netAEorg,netAEdom,aetr_iter,aete_iter,aemodtr_iter,modteX, modteY,cget,nd) #AE with tanh => -1,1

    # Get transformer from original space (not some encoding space) and domain space (not encoded)
    a2tr_iter = cds2([ax0, ax1], ay,shuffle=True)  # Data where training data in orig space is mapped to data in domain space
    a2te_iter = cds2([atex0, atex1], atey)  # modtrX, amodteX = cdsMod(trX), cdsMod(teX)        #eteX, eteY = amodteX[-nFor:], [teY] * nFor    # if cfg["doTra"]:            #ntr=ntra+1 - nFor            #origtr_iter = cds2(amodtrX,np.copy(trY), True)  # Array is copied due to type cast to float32 in cds2            #origte_iter = cds2(amodteX, np.copy(teY), False)
    loadtrans = cget
    orgTransModel, cyccfg, loaded = trainClassifiers.getTrans(cfg, a2tr_iter, a2te_iter, ((ax0, ay), (atex0, atey)), None, loadtrans, selfTra=False)



    #Get labeled domain data, by  applying transformer multiple times on source data
    nFor = len(cfg["transEv"])#   nFor = ntra# if nFor <= 0:        print("nothing to forecast", ntra, cfg["transEv"]) return
    atrX = [ax0]
    atrY = [ay]
    for i in range(nFor):
        cajx, cajy = [], []
        orgTransModel.eval()
        cdataset = cds(atrX[-1], atrY[-1],shuffle=False,norm=cfg["evNorm"]) # citer=cds2([atrX[-2],atrX[-1]],atrY[-1])
        for data in cdataset:
            with tca.autocast():
                dsx = data[0].cuda()
                if not cfg["dirCyc"]: dsx=[None,dsx]
                output =  orgTransModel(dsx).detach().cpu()
                cajx.append(output.clone().numpy())
                cajy.append(data[-1].clone().numpy())
        atrX.append(np.concatenate(cajx, axis=0))
        atrY.append(np.concatenate(cajy, axis=0))
    etrX, etrY = atrX[-nFor:], atrY[-nFor:] #print("nfo", nFor, len(atrX))    # imgutil.makeAndStore2(atrX[-3][:64],atrX[-2][:64],atrX[-1][:64], cfg["bFolder"] + "samples/", "FORCAST"+str(cfg["bid"]) + fs(cfg) + ".png")

    if not failed:
        #Get domain datasets used for prediction
        amodteX,amodteY = cdsMod(teX,teY)
        eteX, eteY = amodteX[-nFor:], amodteY[-nFor:]

        def evalCl(ltrX, ltrY, lteX, lteY,domid):
            def cods(lX, lY, shuffle=False):
                trX, trY = np.concatenate(lX, axis=0), np.concatenate(lY, axis=0)  # trY=np.concatenate([np.ones(aj[0].shape[0],dtype=np.int)*j for j in range(len(aj))])
                trX, trY = sklearn.utils.shuffle(trX, trY)
                trit = cds(trX, trY, shuffle)
                return trit
            trit = cods(ltrX, ltrY, True)
            teit = cods(lteX, lteY, False)
            netCl, clcfg = trainClassifiers.getclassifier(cfg, trit, teit, None, getc=False, save=False, loadCl=False)
            return clcfg

        #Train classifier using labeled data being transformed and apply it to generated test data
        vals=np.arange(nFor)
        for j in reversed(vals):
                clcfg = evalCl([etrX[j]], [etrY[j]], [eteX[j]], [eteY[j]],j)
                #print("eval",j,nFor,np.sum(etrX[j]),clcfg,cyccfg)
                cyccfg["ptrA" + str(j)] = clcfg["trA"]
                cyccfg["pteA" + str(j)] = clcfg["teA"]
                cyccfg["mteA" + str(j)] = clcfg["mteA"]

    cyccfg = {**cyccfg, **othlo,**baseClRes}
    cfg["result"] = [cyccfg]
    print("\n\nBench:",cfg["trans"])
    print("Result")
    res=cfg["result"][0]
    print("All metrics",res)
    print("Accs (Source, target 0,..,2)",np.round([res["teA"]]+[res["pteA"+str(i)] for i in range(3)],3))
    print("MaxAccs", np.round([res["teA"]] + [res["mteA" + str(i)] for i in range(3)], 3))
    print("\n\n\n\n")
    #print("\n\n\n\nFOR Accuracy: check pteA. (=Accuracy after last epoch) and mteA.(=max Accuracy accross all epochs) in results below")
    #print("pteA0 denotes test accuracy on target domain 0, pteA1 target domain 1, pteA2 target domain 2, etc.")
    #print("teA denotes test accuracy on source domain")


