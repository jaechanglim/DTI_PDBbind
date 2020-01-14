import argparse
import utils
import random
random.seed(0)
import numpy as np
from dataset import MolDataset, DTISampler, my_collate_fn
from torch.utils.data import DataLoader                                     
import model 
import os
import torch
import time
import torch.nn as nn
import pickle
from sklearn.metrics import r2_score, roc_auc_score
from collections import Counter
import sys
from scipy import stats
import glob

parser = argparse.ArgumentParser() 
parser.add_argument('--batch_size', help='batch size', type = int, default = 1)
parser.add_argument('--num_workers', help = 'number of workers', type = int, default = 7) 
parser.add_argument('--dim_gnn', help = 'dim_gnn', type = int, default = 32) 
parser.add_argument("--n_gnn", help="depth of gnn layer", type=int, default = 3)
parser.add_argument('--ngpu', help = 'ngpu', type = int, default = 1) 
parser.add_argument('--restart_file', help = 'restart file', type = str) 
parser.add_argument('--filename', help='filename', \
        type = str, default='/home/wykgroup/jaechang/work/ML/PDBbind_DTI/data_pdbbind/pdb_to_affinity.txt')
parser.add_argument('--exp_name', help='experiment name', type=str)
parser.add_argument('--test_output_filename', help='test output filename', type = str, default='test.txt')
parser.add_argument('--key_dir', help='key directory', type = str, default='keys')
parser.add_argument('--data_dir', help='data file path', type = str, \
                    default='/home/udg/msh/urp/DTI_PDBbind/data/')
parser.add_argument("--filter_spacing", help="filter spacing", type=float, default=0.1)
parser.add_argument("--filter_gamma", help="filter gamma", type=float, default=10)
parser.add_argument("--potential", help="potential", type=str, 
                    default='morse_all_pair', 
                    choices=['morse', 'harmonic', 'morse_all_pair'])

args = parser.parse_args()
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
else: 
    print (f'No {args.potential} potential')
    exit(-1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, args.restart_file)
print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

#Dataloader
test_dataset = MolDataset(test_keys, args.data_dir, id_to_y)
test_data_loader = DataLoader(test_dataset, args.batch_size, \
     shuffle=False, num_workers = args.num_workers, collate_fn=my_collate_fn)

#loss
loss_fn = nn.MSELoss()

#test
st = time.time()

test_losses1 = []
test_losses2 = []

test_pred1 = dict()
test_pred2 = dict()
test_true = dict()


model.eval()
for i_batch, sample in enumerate(test_data_loader):
    model.zero_grad()
    if sample is None : continue
    h1, adj1, h2, adj2, A_int, dmv, dmv_rot, valid, affinity, keys = sample
    #print (A_int.sum())
    h1, adj1, h2, adj2, A_int, dmv, dmv_rot, valid, affinity = \
            h1.to(device), adj1.to(device), h2.to(device), adj2.to(device), \
            A_int.to(device), dmv.to(device), dmv_rot.to(device), \
            valid.to(device), affinity.to(device)
    with torch.no_grad():
        pred1 = model(h1, adj1, h2, adj2, A_int, dmv, valid)
        pred2 = model(h1, adj1, h2, adj2, A_int, dmv_rot, valid)
    #print (pred1) 
    affinity = affinity.data.cpu().numpy()
    pred1 = pred1.data.cpu().numpy()
    pred2 = pred2.data.cpu().numpy()
    
    for i in range(len(keys)):
        test_pred1[keys[i]] = pred1[i]
        test_pred2[keys[i]] = pred2[i]
        test_true[keys[i]] = affinity[i]
    #if i_batch>2: break
#Write prediction
loaded_epoch = args.restart_epoch.split("_")[-1].split(".")[0]
w_test = open(os.path.join("output", args.exp_name + "_" + args.test_output_filename.split(".")[0] + "_" + loaded_epoch + args.test_output_filename.split(".")[-1]), 'w')

for k in test_pred1.keys():
    w_test.write(f'{k}\t{test_true[k]:.3f}\t')
    w_test.write(f'{test_pred1[k].sum():.3f}\t')
    for j in range(test_pred1[k].shape[0]):
        w_test.write(f'{test_pred1[k][j]:.3f}\t')
    w_test.write('\n')
w_test.close()

#Cal R2
test_r2 = r2_score([test_true[k].sum(-1) for k in test_true.keys()], \
        [test_pred1[k].sum(-1) for k in test_true.keys()])
slope, intercept, r_value, p_value, std_err = \
        stats.linregress([test_true[k].sum(-1) for k in test_true.keys()],                            
                        [test_pred1[k].sum(-1) for k in test_true.keys()])


end = time.time()
print (f"R2: {test_r2:.3f}")
print (f"R: {r_value:.3f}")
print (f"Time: {end-st:.3f}")
