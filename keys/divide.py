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
random.seed(0)

def count_interactions(fn):
    with open(fn, 'rb') as f:
        m1, m1_uff, m2, interaction_data = pickle.load(f)
    n_int = 0
    for k in interaction_data:
        n_int+=len(interaction_data[k])
    return n_int

with open('/home/wykgroup/jaechang/work/data/CASF-2013/coreset/index/2013_core_data.lst') as f:
    lines = f.readlines()[5:]
    casf_2013 = [l.split()[0] for l in lines]

casf_2016 = glob.glob('/home/jaechang/work/jaechang_horus/work/data/CASF-2016/coreset/*')
casf_2016 = [c.split('/')[-1] for c in casf_2016] 

keys = glob.glob('../../data_pdbbind2/data/*')[:]
keys = [k for k in keys if count_interactions(k)>0]
keys = [k.split('/')[-1] for k in keys]
#print (keys)
with open('../../data_pdbbind2/train_pdbs.txt') as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines]
    train_keys = sorted(list( set(keys) & set(lines)))
    train_keys = [k for k in train_keys if k not in casf_2013]
    train_keys = [k for k in train_keys if k not in casf_2016]

with open('../../data_pdbbind2/test_pdbs.txt') as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines]
    test_keys = sorted(list( set(keys) & set(lines)))



print (f'Number of train data: {len(train_keys)}')
print (f'Number of test data: {len(test_keys)}')
with open('train_keys.pkl', 'wb') as fp:
    pickle.dump(train_keys, fp)
with open('test_keys.pkl', 'wb') as fp:
    pickle.dump(test_keys, fp)

