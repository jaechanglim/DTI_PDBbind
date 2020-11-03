import glob
import random
import os
import pickle

keys = glob.glob("../data/????_*")
keys = [k.split("/")[-1] for k in keys]
keys = sorted(keys, key=lambda k: (k.split("_")[0], int(k.split("_")[1])))
with open(f"./test_keys.pkl", "wb") as fp:
    pickle.dump(keys, fp)
