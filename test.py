import os
import random
import sys
import glob
import argparse
import time
from collections import Counter

import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import arguments
import utils
from dataset import MolDataset
from dataset import DTISampler
from dataset import tensor_collate_fn
import model

random.seed(0)
args = arguments.parser(sys.argv)
print(args)

# Read labels
with open(args.filename) as f:
    lines = f.readlines()
    lines = [l.split() for l in lines]
    id_to_y = {l[0]: float(l[1]) for l in lines}

with open(args.key_dir+"/test_keys.pkl", "rb") as f:
    test_keys = pickle.load(f)


# Model
cmd = utils.set_cuda_visible_device(args.ngpu)
os.environ["CUDA_VISIBLE_DEVICES"] = cmd[:-1]
if args.potential == "morse":
    model = model.DTILJ(args)
elif args.potential == "morse_all_pair":
    model = model.DTILJAllPair(args)
elif args.potential == "harmonic":
    model = model.DTIHarmonic(args)
elif args.potential == "gnn":
    model = model.GNN(args)
elif args.potential == "cnn3d":
    model = model.CNN3D(args)
elif args.potential == "cnn3d_kdeep":
    model = model.CNN3D_KDEEP(args)
else:
    print(f"No {args.potential} potential")
    exit(-1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, args.restart_file)

print(f"vina_hbond_coeff: {model.vina_hbond_coeff.data.cpu().numpy()[0]:.3f}")
print(f"vina_hydrophobic_coeff: \
{model.vina_hydrophobic_coeff.data.cpu().numpy()[0]:.3f}")
print(f"rotor_coeff: {model.rotor_coeff.data.cpu().numpy()[0]:.3f}")
print(f"vdw_coeff: {model.vdw_coeff.data.cpu().numpy()[0]:.3f}")
# exit(-1)
print("number of parameters : ",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

# Dataloader
test_dataset = MolDataset(test_keys, args.data_dir, id_to_y)
test_data_loader = DataLoader(test_dataset, args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              collate_fn=tensor_collate_fn)

# test
st = time.time()

test_losses1 = []
test_losses2 = []

test_pred = dict()
test_true = dict()

if args.with_uncertainty:
    epi_var_dict = dict()
    ale_var_dict = dict()
    tot_var_dict = dict()

model.eval()
for i_batch, sample in enumerate(test_data_loader):
    model.zero_grad()
    if sample is None:
        continue
    sample = utils.dic_to_device(sample, device)
    keys = sample["key"]
    affinity = sample["affinity"]

    if args.with_uncertainty:
        MC_component_pred = []
        ale_var = []
    with torch.no_grad():
        if args.with_uncertainty:  # with uncertainty
            for i in range(args.n_mc_sampling):
                pred = model(sample)[0]
                var = model(sample)[-1]
                MC_component_pred.append(pred.dta.cpu().numpy())
                ale_var.append(var.data.cpu().numpy())
            MC_component_pred = np.array(MC_component_pred)
            ale_var = np.mean(np.array(ale_var), axis=0)
        else:  # without uncertainty
            pred = model(sample)[0]
            pred = pred.data.cpu().numpy()
        affinity = affinity.data.cpu().numpy()

    for i in range(len(keys)):
        key = keys[i]
        test_pred[key] = pred[i]  # True energy
        test_true[key] = affinity[i]
        if args.with_uncertainty:
            MC_pred_i = MC_component_pred[i].sum(-1)
            test_pred[key] = np.mean(MC_pred_i, axis=0)  # predicted energy
            epi_var_dict[key] = np.var(MC_pred_i, axis=0)  #epistemic variance
            ale_var_dict[key] = ale_var[i]  #a aleatoric variance
            # Total variance
            tot_var_dict[key] = epi_var_dict[key] + ale_var_dict[key]

# Compute metrics
if args.with_uncertainty:
    true_list = np.array([test_true[k].sum(-1) for k in test_true.keys()])
    pred_list = np.array([test_pred[k].sum(-1) for k in test_true.keys()])
    test_r2 = r2_score(true_list, pred_list)
    test_mse = mean_squared_error(true_list, pred_list)
    _, _, r_value, _, _ = stats.linregress(true_list, pred_list)
    end = time.time()
else:
    test_r2 = r2_score([test_true[k].sum(-1) for k in test_true.keys()],
                       [test_pred[k].sum(-1) for k in test_true.keys()])
    slope, intercept, r_value, p_value, std_err = \
        stats.linregress([test_true[k].sum(-1) for k in test_true.keys()],
                         [test_pred[k].sum(-1) for k in test_true.keys()])
end = time.time()

# Write prediction
w_test = open(args.test_result_filename, "w")
for k in sorted(test_pred.keys()):
    w_test.write(f"{k}\t{test_true[k]:.3f}\t")
    w_test.write(f"{test_pred[k].sum():.3f}\t")
    if args.with_uncertainty:
        # variance for predicting total energy
        w_test.write(f"{epi_var_dict[k]:.5f}\t")
        w_test.write(f"{ale_var_dict[k]:.7f}\t")
        w_test.write(f"{tot_var_dict[k]:.5f}\t")
    else:
        for j in range(test_pred[k].shape[0]):
            w_test.write(f"{test_pred[k][j]:.3f}\t")
    w_test.write("\n")
w_test.close()

# Cal R2
if args.with_uncertainty:
    print(f"MSE: {test_mse:.3f}\n")
print(f"R2: {test_r2:.3f}")
print(f"R: {r_value:.3f}")
print(f"Time: {end-st:.3f}")
