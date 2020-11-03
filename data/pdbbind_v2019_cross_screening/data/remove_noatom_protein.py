import pickle
import glob
import os
import multiprocessing
from multiprocessing import Pool


def check(k):
    with open(k, "rb") as f:
        _, _, p, _ = pickle.load(f)
    if p.GetNumAtoms() == 0:
        os.remove(k)
        print(k)

total = glob.glob("????_????")
pool = Pool(20)
r = pool.map_async(check, total)
r.wait()
pool.close()
pool.join()
