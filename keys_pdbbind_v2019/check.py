import pickle
import os
test_keys = pickle.load(open('../keys/test_keys.pkl', 'rb'))

for k in test_keys:
    if not os.path.exists(f'../../data_pdbbind_v2019/data/{k}'):
        print (k)
