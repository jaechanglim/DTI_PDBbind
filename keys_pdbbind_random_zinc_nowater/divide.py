import pickle
from collections import Counter
import random
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
from rdkit import Chem
import numpy as np
import glob
import gc
import time
import sys
import os
random.seed(0)
#train and test pdbs
pdbbind_train_keys = pickle.load(open(f'../keys_pdbbind_v2019/train_keys.pkl', 'rb'))
pdbbind_test_keys = pickle.load(open(f'../keys_pdbbind_v2019/test_keys.pkl', 'rb'))

#zinc
zinc_filenames = glob.glob('/home/wykgroup/jaechang/work/data/DUD-E/all/*/decoys_final.ism')
zinc = []
for fn in zinc_filenames: zinc += open(f'{fn}').readlines()
zinc = set(['ZINC0000'+z.strip().split()[1][1:] for z in zinc])

#choose train keys
keys = glob.glob('../../data_pdbbind_random_zinc_nowater/data/*')[:]
keys = [k.split('/')[-1] for k in keys]
train_keys = [k for k in keys if k.split('_')[0] in pdbbind_train_keys] 
train_keys = [k for k in train_keys if k.split('_')[1] not in zinc]
train_mol_ids = set([k.split('_')[1] for k in train_keys])

#choose test keys
test_keys = [k for k in keys if k.split('_')[0] in pdbbind_test_keys]
test_keys = [k for k in test_keys if k.split('_')[1] not in train_mol_ids]

print (len(train_keys), len(test_keys))
with open(f'train_keys.pkl', 'wb') as fp:
    pickle.dump(train_keys, fp)
with open(f'test_keys.pkl', 'wb') as fp:
    pickle.dump(test_keys, fp)
