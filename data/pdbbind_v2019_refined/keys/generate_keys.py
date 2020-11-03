import glob
import random
import pickle

total = glob.glob("../data/????")
test = glob.glob("../pp_test/????")
total = [t.split("/")[-1] for t in total]
test = [t.split("/")[-1] for t in test]
train = list(set(total) - set(test))
with open("train_keys.pkl", "wb") as w:
    pickle.dump(train, w)
with open("test_keys.pkl", "wb") as w:
    pickle.dump(test, w)

