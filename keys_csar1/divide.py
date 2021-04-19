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

keys = glob.glob('../../data_csar_nrc_hiq_set1/data/*')[:]
keys = [k.split('/')[-1] for k in keys]
#print (keys)
with open('pdbs.txt') as f:
    lines = f.readlines()[0].split()

test_keys = sorted(list(set(keys) & set(lines)))
print (len(test_keys))
with open('test_keys.pkl', 'wb') as fp:
    pickle.dump(test_keys, fp)
