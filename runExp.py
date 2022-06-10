#Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra
#Licence: Use it however you like, but cite the paper :-)

print("Source code for 'Domain Transformer: Predicting Samples of Unseen, Future Domains' by Johannes Schneider, IJCNN, 2022, https://arxiv.org/abs/2106.06057;  Github; https://github.com/JohnTailor/DoTra")
print("Note: You should do at least 12 runs and only take those accuracies, and report the quartile, i.e., the 4th best result, due to instability of Cycle GAN (as described in paper)")

from main import trainOne

#Get DTtra for rotation, zoom and splitting for MNIST

dummy = True #For checking if runs
dummy = False

baseCfg={'imSi':32,'benchID': 0, 'netSi': 1.0, 'ds': ('MNIST', 10, 60000 if not dummy else 2000),'distinctDat': 1, 'datCut': 1, 'batchSize': 64,'dummy': dummy,
         'addMin': 1, 'baseZoom': 1, 'aez': 32, 'aeType': 'N', 'aebn': 1, 'aebeta': 1,
         'aecfg': {'id': 0, 'opt': ('A', 0.001, 1e-05), 'netT': 'V6'}, 'aeGAN': (0, {'lr': 0.0001}), 'aeganbn': 1, 'sepAE': 1,'clcfg': {'id': 0, 'opt': ('S', 0.1, 0.0005), 'netT': 'V6'}, 'addM': 1,'ttry': (3, 0.3), 'resB': 0,
        'convdim': 64, 'nExLay': 0, 'tleak': 0.005, 'epT': 96 if not dummy else 3, 'trcfg': {'id': 0, 'opt': ('A', 0.001, 1e-05), 'netT': 'V6'}, 'epA': 72 if not dummy else 3, 'ker': 3, 'twoLin': 0, 'drop': 0, 'bn': 1, 'epC': 36 if not dummy else 3, 'evNorm': 1,  'genAE': 0, 'singleIn': 1,'traLo': 2, 'sym': 0,
         'selfAE': 0,
         'dlay': [128], 'glay': [32], 'laLeak': 1, 'gben': 2, 'dben': 2, 'gdrop': (0, 0.2, 0), 'ddrop': (0, 0.2, 0), 'useLab': 0, 'useRec': 3, 'recLo': 1, 'cycFac': [1, 1], 'solvSim': 0, 'd1': 1, 'labSmo': 0.1, 'DGlr': [0.0001, 0.0001], 'DGBeta12': ((0.5, 0.999), (0.5, 0.999)), 'lrdecay': 0.66, 'ntries': (200, 3), 'epG': 96 if not dummy else 4, 'useLat': 0, 'llay': [8, 8], 'ldrop': (0, 0.1, 0, 0), 'lben': 1,  'mlay': [4, 4], 'mdrop': (0, 0, 0), 'mben': 0, 'mLR': 0.001, 'epM': 144 if not dummy else 3, 'mlrdecay': 0.8,
         'mBatchSize': 512, 'mntries': (200, 3), 'useClLo': 0, 'trainCl': 0, 'trainOrg': 0, 'smoo': 0, 'useDiff': 0, 'cllay': [4, 4], 'clben': 2, 'cldrop': [0, 0, 0.1], 'epLin': 24, 'filterCl': 0, 'dirCyc': 0, 'accStop': (0, 0.5),
         'bpart': '', 'disDat': 1, 'dt':'None'
         }

cfg1={**baseCfg,**{'trans': [('splitMid1', 6)],  'transEv': [('splitMid1', 6), ('splitMid1', 12), ('splitMid1', 18)],  'pr': {'dt':'None'}}}
cfg2={**baseCfg,**{'trans': [('zoo1', 1.33)],    'transEv': [('zoo1', 1.33), ('zoo1', 1.769), ('zoo1', 2.353)],'pr': {'dt':'None'} }}
cfg3={**baseCfg,**{'trans': [('rot1', 45)],  'transEv': [('rot1', 45), ('rot1', 90), ('rot1', 135)],'pr': {'dt':'None'}}}

cfgs=[cfg1,cfg2,cfg3]
for i in range(12):
    for c in cfgs:
        print("\n\nRUN ",i, " Bench",c["trans"])
        c["dummy"]=dummy
        trainOne(c)