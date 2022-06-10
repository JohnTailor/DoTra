#Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
#Licence: Use it however you like, but cite the paper :-)

from scipy import ndimage
from torch.utils.data import Dataset
import numpy as np,os,sklearn #,pickle imageio,time,
import torchvision,torch
import torchvision.transforms as transforms


def getnorm(dname):
    if dname == "MNIST":
        return (torch.from_numpy(np.array((0.1307), np.float32).reshape(1, 1, 1, 1)).cuda(), torch.from_numpy(np.array((0.3081), np.float32).reshape(1, 1, 1, 1)).cuda())


def getFullDS(cfg,ntrain,sname,cget):
    dname=cfg["ds"][0]
    trans=transforms.Compose([transforms.ToTensor()])
    if dname == "MNIST":
        cdat = torchvision.datasets.MNIST
        cfg["imCh"] = 1
    down=True
    cpa="."
    fname="Mnist"
    if not os.path.exists(fname+"teX") or cget:
       os.makedirs(cpa,exist_ok=True)
       def loadStore(isTrain,ndat):
            trainset = cdat(root=".", train=isTrain, download=down,transform=trans)
            train_dataset = torch.utils.data.DataLoader(trainset, batch_size=ndat, num_workers=0)  # cfg["batchSize"]
            ds = next(iter(train_dataset))
            X,y=ds[0].clone().numpy(),ds[1].clone().numpy()
            print("Data stats",dname,X.shape,np.mean(X,axis=(0,2,3)),np.std(X,axis=(0,2,3)))
            if (dname == "MNIST" or dname == "Fash") and cfg["imSi"]!=28:
                X=[ndimage.zoom(X[i,0],cfg["imSi"]/28) for i in range(X.shape[0])]
                X=np.stack(X,axis=0)
                X=np.expand_dims(X,axis=1)
            ds = [X, y]
            ds = sklearn.utils.shuffle(*ds)  # , random_state=cfg["seed"])
            t=np.float16
            preamb="tr" if isTrain else "te"
            with open(fname + preamb+"X", "wb") as f: np.save(f, ds[0].astype(t), allow_pickle=True)
            with open(fname + preamb+"Y", "wb") as f: np.save(f, ds[1].astype(np.int16), allow_pickle=True)
            #return trainset
       loadStore(True,ntrain)
       loadStore(False, ntrain)
    lo = lambda na: np.load(open(fname + na, "rb"), allow_pickle=True)
    trX,trY=lo("trX"),lo("trY")
    teX,teY=lo("teX"),lo("teY")

    norm=getnorm(dname)
    trX = (trX - norm[0].cpu().numpy()) / norm[1].cpu().numpy()
    teX = (teX - norm[0].cpu().numpy()) / norm[1].cpu().numpy()
    return (trX, trY), (teX, teY)#, None,norm
