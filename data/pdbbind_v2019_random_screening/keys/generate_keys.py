import glob
import random
import pickle


total = glob.glob("../data/*_????")
total = [t.split("/")[-1] for t in total]
random.shuffle(total)
length = len(total)
train = total[length//13:]
test = total[:length//13]
with open("train_keys.pkl", "wb") as w:
    pickle.dump(train, w)
with open("test_keys.pkl", "wb") as w:
    pickle.dump(test, w)
