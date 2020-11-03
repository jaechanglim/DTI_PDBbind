import glob
import random
import os
import pickle

keys = glob.glob("../data/????")
keys = [k.split("/")[-1] for k in keys]
keys = sorted(keys)
with open(f"./test_keys.pkl", "wb") as fp:
    pickle.dump(keys, fp)
