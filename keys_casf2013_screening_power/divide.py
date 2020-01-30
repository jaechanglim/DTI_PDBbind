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

keys = glob.glob('../../data_casf2013_decoy_screening/data/*')[:]
keys = [k.split('/')[-1] for k in keys]
#print (keys)

test_keys = sorted(keys[:])
print (len(test_keys))
with open('test_keys.pkl', 'wb') as fp:
    pickle.dump(test_keys, fp)
