import glob
import random
import os
import pickle

keys = glob.glob("../data/????_????_*")
keys = [k.split("/")[-1] for k in keys]
random.shuffle(keys)
n = 100
interval = len(keys) // n
for i in range(n):
    os.makedirs(f"{i}", exist_ok=True)
    st = i * interval
    end = (i + 1) * interval
    if i == n - 1:
        end = None
    with open(f"{i}/test_keys.pkl", "wb") as fp:
        pickle.dump(keys[st:end], fp)
