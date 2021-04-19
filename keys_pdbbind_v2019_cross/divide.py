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
pdbbind_train_keys = pickle.load(open(f'../keys_pdbbind_v2019/train_keys.pkl', 'rb'))
pdbbind_test_keys = pickle.load(open(f'../keys_pdbbind_v2019/test_keys.pkl', 'rb'))
keys = glob.glob('../../data_pdbbind_v2019_cross/data/*')[:]
keys = [k.split('/')[-1] for k in keys]
train_keys = [k for k in keys if k.split('_')[1] in pdbbind_train_keys and 
                    k.split('_')[0] in pdbbind_train_keys] 
test_keys = [k for k in keys if k.split('_')[1] in pdbbind_test_keys
                and k.split('_')[1] in pdbbind_test_keys] 
print (len(train_keys), len(test_keys))
with open(f'train_keys.pkl', 'wb') as fp:
    pickle.dump(train_keys, fp)
with open(f'test_keys.pkl', 'wb') as fp:
    pickle.dump(test_keys, fp)
