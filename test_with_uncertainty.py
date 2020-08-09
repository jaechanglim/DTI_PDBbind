import os
import sys
import time

import numpy as np
import pickle
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MolDataset, DTISampler, tensor_collate_fn
import arguments
import utils
import model


args = arguments.parser(sys.argv)
print (args)

#Read labels
with open(args.filename) as f:
    lines = f.readlines()
    lines = [l.split() for l in lines]
    id_to_y = {l[0]:float(l[1]) for l in lines}

with open(args.key_dir+'/test_keys.pkl', 'rb') as f:
    test_keys = pickle.load(f)


#Model
cmd = utils.set_cuda_visible_device(args.ngpu)
os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]

if args.potential=='morse': model = model.DTILJ(args)
elif args.potential=='morse_all_pair': model = model.DTILJAllPair(args)
elif args.potential=='harmonic': model = model.DTIHarmonic(args)
elif args.potential=='harmonic_interaction_specified': model = model.DTIHarmonicIS(args)
else:
    print (f'No {args.potential} potential')
    exit(-1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = utils.initialize_model(model, device, load_save_file=True, file_path=args.restart_file)
print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

#Dataloader
test_dataset = MolDataset(test_keys, args.data_dir, id_to_y)
test_data_loader = DataLoader(test_dataset, args.batch_size, \
     shuffle=False, num_workers = args.num_workers, collate_fn=tensor_collate_fn)
# ========================================================================================
st = time.time()

affinity_dict = dict()
pred_dict = dict()

epi_var_dict = dict()
ale_var_dict = dict()
tot_var_dict = dict()

model.eval()
for i_batch, sample in enumerate(test_data_loader):
    model.zero_grad()
    if sample is None : continue
    sample = utils.dic_to_device(sample, device)
    keys = sample['key']
    affinity = sample['affinity']

    MC_component_pred = []
    ale_var = []
    with torch.no_grad():
        for i in range(args.n_mc_sampling):
            pred, _, _, var = model(sample)
            MC_component_pred.append(pred.data.cpu().numpy())
            ale_var.append(var.data.cpu().numpy())

        MC_component_pred = np.array(MC_component_pred)  # [n_key, n_mc_sampling, n_energy_component, ]
        ale_var = np.mean(np.array(ale_var), axis=0)
        affinity = affinity.data.cpu().numpy()

    for i in range(len(keys)):
        key = keys[i]
        affinity_dict[key] = affinity[i]  # True energy
        MC_pred_i = MC_component_pred[i].sum(-1)   # [n_mc_sampling,]
        pred_dict[key] = np.mean(MC_pred_i, axis=0)  # predicted energy
        epi_var_dict[key] = np.var(MC_pred_i, axis=0)  # epistemic variance
        ale_var_dict[key] = ale_var[i]  # aleatoric variance
        tot_var_dict[key] = epi_var_dict[keys[i]] + ale_var_dict[keys[i]]  # total variance

#Compute metrics
key_list = list(affinity_dict.keys())
affinity_list = np.array([affinity_dict[k].sum(-1) for k in affinity_dict.keys()])
pred_list = np.array([pred_dict[k].sum(-1) for k in affinity_dict.keys()])
test_r2 = r2_score(affinity_list, pred_list)
test_mse = mean_squared_error(affinity_list, pred_list)
_, _, r_value, _, _ = stats.linregress(affinity_list, pred_list)
end = time.time()

#Write prediction
w_test = open(args.test_result_filename, 'w')
for k in sorted(key_list):
    w_test.write(f'{k}\t{affinity_dict[k]:.5f}\t')  # Key, True value
    w_test.write(f'{pred_dict[k]:.5f}\t')   # predicted total energy
    w_test.write(f'{epi_var_dict[k]:.5f}\t')  # variance for predicting total energy
    w_test.write(f'{ale_var_dict[k]:.7f}\t')  # variance for predicting total energy
    w_test.write(f'{tot_var_dict[k]:.5f}\t')  # variance for predicting total energy
    w_test.write('\n')
w_test.close()

print (f"R2: {test_r2:.3f}\n")
print (f"MSE: {test_mse:.3f}\n")
print (f"R: {r_value:.3f}\n")
print (f"Time: {end-st:.3f}")