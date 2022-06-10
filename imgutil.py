#Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
#Licence: Use it however you like, but cite the paper :-)

import numpy as np
from scipy import ndimage

def nor(x): return (x - np.mean(x, axis=(0, 2, 3), keepdims=True)) / (np.std(x, axis=(0, 2, 3), keepdims=True) + 1e-7)


def rot1(img, d):
    #co = np.min(img) #img[ 0, 0]
    co=min(img[0,0],img[-1,-1])
    #print(np.min(img),np.max(img),np.mean(img),np.std(img),np.median(img),"test")
    img = ndimage.rotate(img, d, reshape=False,cval=co)
    #print(co, img[0, 0],bef, "rotting")
    return img



def zoo1(img, d):
    bo = int(32 * d - 32) // 2
    img = ndimage.zoom(img, d)[bo:32 + bo, bo:32 + bo]
    return img


def splitMid1(x, d):
    w=x.shape[0]
    left=x[d//2:w//2,:]
    mid=x[:d]*0+min(x[0,0],x[-1,-1])
    right = x[ w//2:, :]
    x=np.concatenate([left,mid,right],axis=0)[:w]
    return x

def getOp(name):
    return globals()[name]

def applyOp(x,opname,para):
    x=x.astype(np.float32)
    op=getOp(opname)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):  # xo=np.copy(x[i,j])
            x[i,j]=op(x[i,j],para)
    return nor(x)


