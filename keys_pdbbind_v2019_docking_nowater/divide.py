import glob
import pickle

train_keys = pickle.load(open('../keys_pdbbind_v2019/train_keys.pkl', 'rb'))
test_keys = pickle.load(open('../keys_pdbbind_v2019/test_keys.pkl', 'rb'))

keys = glob.glob('/home/wykgroup/udg/mseok/data/data_pdbbind_v2016_docking_nowater/data/*')
keys = [k.split('/')[-1] for k in keys]
train_keys = [k for k in keys if k.split('_')[0] in train_keys]
test_keys = [k for k in keys if k.split('_')[0] in test_keys]
print (len(train_keys))
print (len(test_keys))
with open('train_keys.pkl', 'wb') as fp:
    pickle.dump(train_keys, fp)
with open('test_keys.pkl', 'wb') as fp:
    pickle.dump(test_keys, fp)
