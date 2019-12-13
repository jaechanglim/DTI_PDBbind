import argparse
import utils
import random
random.seed(0)
import numpy as np
from dataset import MolDataset, DTISampler, my_collate_fn
from torch.utils.data import DataLoader                                     
import os
import torch
import time
import torch.nn as nn
import pickle
from sklearn.metrics import r2_score, roc_auc_score
from collections import Counter
import sys
import glob
from model import mpnn

parser = argparse.ArgumentParser() 
parser.add_argument('--lr', help="learning rate", type=float, default = 1e-4)
parser.add_argument("--lr_decay", help="learning rate decay", type=float, default=1.0)
parser.add_argument("--weight_decay", help="weight decay", type=float, default = 0.0)
parser.add_argument('--num_epochs', help='number of epochs', type = int, default = 100)
parser.add_argument('--batch_size', help='batch size', type = int, default = 1)
parser.add_argument('--num_workers', help = 'number of workers', type = int, default = 7) 
parser.add_argument('--ngpu', help = 'ngpu', type = int, default = 1) 
parser.add_argument('--save_dir', help = 'save directory', type = str) 
parser.add_argument('--filename', help='filename', type = str, default='../data/data.txt')
parser.add_argument('--key_dir', help='key directory', type = str, default='keys')
parser.add_argument('--data_dir', help='data file path', type = str, default='../data_chembl_nn2/data/')
parser.add_argument('--restart_file', help='restart file', type = str)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.3)
parser.add_argument("--active_upsampling_ratio", help="active upsampling ratio", type=float, default=1.0)
parser.add_argument("--output_filename", help="output filename", type=str, default='test.txt')
parser.add_argument('--dim_gnn', help = 'dim_gnn', type = int, default = 32) 
parser.add_argument("--n_gnn", help="depth of gnn layer", type=int, default = 3)
parser.add_argument("--filter_spacing", help="filter spacing", type=float, default=0.1)
parser.add_argument("--filter_gamma", help="filter gamma", type=float, default=10)
parser.add_argument("--random_rotation", help="random rotation", type=float, default = 0.0)

args = parser.parse_args()
print (args)


with open(args.filename) as f:
    lines = f.readlines()
    lines = [l.split() for l in lines]
    lines = [l for l in lines if not np.isnan(float(l[-3])) and not np.isinf(float(l[-3]))]
    id_to_y = {l[0]+'_'+l[1]+'_'+l[-2]:float(l[-3]) for l in lines}

with open(args.key_dir+'/test_uniprots.pkl', 'rb') as f:
    test_uniprots = pickle.load(f)

with open(args.key_dir+'/test_keys.pkl', 'rb') as f:
    test_keys = pickle.load(f)
    test_keys = [k for k in test_keys if k in id_to_y]


#random.shuffle(train_keys)
#print ('number of train uniprot: ', len(train_uniprots))
#print ('number of test uniprot: ', len(test_uniprots))
print ('number of test active_data: ', len([k for k in test_keys if id_to_y[k]>6]))
print ('number of test inactive_data: ', len([k for k in test_keys if id_to_y[k]<5]))

test_keys = [k for k in test_keys if id_to_y[k]<5 or id_to_y[k]>6]

cmd = utils.set_cuda_visible_device(args.ngpu)
os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]

model = mpnn(args)
#classifier = PointNetCls(1, args.feature_transform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, args.restart_file)

print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

def to_gpu(tensor_list, device):
    return [t.to(device) for t in tensor_list]

test_dataset = MolDataset(test_keys, args.data_dir, id_to_y, args.random_rotation)
test_data_loader = DataLoader(test_dataset, args.batch_size, \
     shuffle=True, num_workers = args.num_workers, collate_fn=my_collate_fn)
loss_fn = nn.MSELoss()

test_losses = []
test_pred = dict()
test_true = dict()

w = open(args.output_filename, 'w')
model.eval()
st = time.time()
for i_batch, sample in enumerate(test_data_loader):

    if sample is None : continue
    H1, A1, H2, A2, DM, VDM, V, Y, keys = sample

    H1, A1, H2, A2, DM, VDM, V, Y = \
            H1.to(device), A1.to(device), H2.to(device), A2.to(device), \
            DM.to(device), VDM.to(device), V.to(device), Y.to(device)
    

    #if H.size()[1]>95: continue
    with torch.no_grad():
        pred_Y = model(H1, A1, H2, A2, DM, VDM, V).squeeze(-1)
    pred_Y = pred_Y.squeeze(-1)
    loss = loss_fn(pred_Y, Y)
    test_losses.append(loss.data.cpu().numpy())
    Y = Y.data.cpu().numpy()
    pred_Y = pred_Y.data.cpu().numpy()
    for i in range(len(keys)):
        w.write(f'{keys[i]} {Y[i]} {pred_Y[i]}\n')
    for i in range(len(keys)):
        test_pred[keys[i]] = pred_Y[i]
        test_true[keys[i]] = Y[i]
    #if i_batch>2 : break
w.close()
test_losses = np.mean(np.array(test_losses))
test_auroc = []
for u in test_uniprots:
    keys = [k for k in test_pred.keys() if u in k]
    #print (keys)
    true = [test_true[k] for k in keys]
    pred = [test_pred[k] for k in keys]
    try:
        test_auroc.append(r2_score(true, pred))
    except:
        continue

test_auroc = np.mean(np.array(test_auroc))
end = time.time()
print ("%.3f\t%.3f\t%.3f" \
    %(test_losses, test_auroc, end-st))
    
    
