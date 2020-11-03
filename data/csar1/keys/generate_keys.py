import pickle
import glob

total = glob.glob("../data/????")
total = [t.split("/")[-1] for t in total]
with open("./test_keys.pkl", "wb") as w:
    for t in total:
        w.write(f"{t}\n")
