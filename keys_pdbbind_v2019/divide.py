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
import codecs
random.seed(0)

def count_interactions(fn):
    with open(fn, 'rb') as f:
        m1, m1_uff, m2, interaction_data = pickle.load(f)
    n_int = 0
    for k in interaction_data:
        n_int+=len(interaction_data[k])
    return n_int

#casf_2013
with open('/home/wykgroup/jaechang/work/data/CASF-2013/coreset/index/2013_core_data.lst') as f:
    lines = f.readlines()[5:]
    casf_2013 = [l.split()[0] for l in lines]

#casf_2016
casf_2016 = glob.glob('/home/wykgroup/jaechang/work/data/CASF-2016/coreset/*')
casf_2016 = [c.split('/')[-1] for c in casf_2016] 

#csar
with codecs.open('/home/wykgroup/jaechang/work/data/CSAR_NRC_HiQ_Set/CSAR_NRC_HiQ_Set.csv', 'r', encoding='utf-8',
                 errors='ignore') as f: 
    lines = f.readlines()[2:]
    csar = [l.split()[0] for l in lines]

keys = glob.glob('../../data_pdbbind_v2019/data/*')[:]
keys = [k.split('/')[-1] for k in keys]
#with open('../../data_pdbbind2/test_pdbs.txt') as f:
#    lines = f.readlines()
#    lines = [l.strip() for l in lines]
#    test_keys = sorted(list( set(keys) & set(lines)))

test_keys = sorted(list( set(keys) & set(casf_2016)))
train_keys = [k for k in keys if k not in test_keys]
train_keys = [k for k in train_keys if k not in casf_2013]
train_keys = [k for k in train_keys if k not in casf_2016]
train_keys = [k for k in train_keys if k not in csar]

print (f'Number of train data: {len(train_keys)}')
print (f'Number of test data: {len(test_keys)}')
print (f'Number of csar: {len(csar)}')
print (f'Number of casf 2013: {len(casf_2013)}')
print (f'Number of casf 2016: {len(casf_2016)}')
with open('train_keys.pkl', 'wb') as fp:
    pickle.dump(train_keys, fp)
with open('test_keys.pkl', 'wb') as fp:
    pickle.dump(test_keys, fp)

