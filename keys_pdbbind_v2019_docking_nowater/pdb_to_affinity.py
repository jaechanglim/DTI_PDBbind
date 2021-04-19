import glob
import pickle

with open('../../data_pdbbind_v2019/pdb_to_affinity.txt') as f:
    lines = f.readlines()
    lines = [l.split() for l in lines]
    pdb_to_affinity = {l[0]:l[1] for l in lines}
keys = []
keys+=pickle.load(open('train_keys.pkl', 'rb'))
keys+=pickle.load(open('test_keys.pkl', 'rb'))
keys = sorted(keys)
with open('pdb_to_affinity.txt', 'w') as w:
    for k in keys:
        w.write(f"{k}\t{pdb_to_affinity[k.split('_')[0]]}\n")
