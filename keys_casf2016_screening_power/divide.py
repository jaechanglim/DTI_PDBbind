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
import os

keys = glob.glob('../../data_casf2016_decoy_screening/data/*')[:]
keys = [k.split('/')[-1] for k in keys]
random.shuffle(keys)
n = 100
interval = len(keys)//n
for i in range(n):
    os.makedirs(f'{i}', exist_ok=True)
    st = i*interval
    end = (i+1)*interval
    if i==n-1: end=-1
    with open(f'{i}/test_keys.pkl', 'wb') as fp:
        pickle.dump(keys[st:end], fp)
