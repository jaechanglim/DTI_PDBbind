import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import time
import torch.nn as nn
import pickle
from sklearn.metrics import r2_score, roc_auc_score
from scipy import stats

import utils
import model 
from dataset import MolDataset, DTISampler, my_collate_fn

parser = argparse.ArgumentParser() 
parser.add_argument('--lr', help="learning rate", type=float, default=1e-4)
parser.add_argument("--lr_decay", help="learning rate decay", type=float, default=1.0)
parser.add_argument("--weight_decay", help="weight decay", type=float, default=0.0)
parser.add_argument('--num_epochs', help='number of epochs', type=int, default=100)
parser.add_argument('--batch_size', help='batch size', type=int, default=1)
parser.add_argument('--num_workers', help='number of workers', type=int, default=7) 
parser.add_argument('--dim_gnn', help='dim_gnn', type=int, default=32) 
parser.add_argument("--n_gnn", help="depth of gnn layer", type=int, default=3)
parser.add_argument('--ngpu', help='ngpu', type=int, default=1) 
parser.add_argument('--save_dir', help='save directory', type=str)
parser.add_argument('--exp_name', help='experiment name', type=str)
parser.add_argument('--restart_file', help='restart file', type=str) 
parser.add_argument('--filename', help='filename', \
parser.add_argument('--train_output_filename', help='train output filename', type=str, default='train.txt')
parser.add_argument('--test_output_filename', help='test output filename', type=str, default='test.txt')
parser.add_argument('--key_dir', help='key directory', type=str, default='keys')
parser.add_argument('--data_dir', help='data file path', type=str, \
                    default='/home/udg/msh/urp/DTI_PDBbind/data/')
parser.add_argument("--filter_spacing", help="filter spacing", type=float, default=0.1)
parser.add_argument("--filter_gamma", help="filter gamma", type=float, default=10)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.0)
parser.add_argument("--loss2_ratio", help="loss2 ratio", type=float, default=1.0)
parser.add_argument("--potential", help="potential", type=str, 
                    default='morse_all_pair', 
                    choices=['morse', 'harmonic', 'morse_all_pair'])
args = parser.parse_args()
print (args)

#Make directory for save files
os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)

#Read labels
with open(args.filename) as f:
    lines = f.readlines()
    lines = [l.split() for l in lines]
    id_to_y = {l[0]:float(l[1]) for l in lines}

with open(args.key_dir+'/train_keys.pkl', 'rb') as f:
    train_keys = pickle.load(f)
with open(args.key_dir+'/test_keys.pkl', 'rb') as f:
    test_keys = pickle.load(f)


#Model
cmd = utils.set_cuda_visible_device(args.ngpu)
os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]

if args.potential=='morse': model = model.DTIMorse(args)
elif args.potential=='morse_all_pair': model = model.DTIMorseAllPair(args)
elif args.potential=='harmonic': model = model.DTIHarmonic(args)
else: 
    print (f'No {args.potential} potential')
    exit(-1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, args.restart_file)

print ('number of parameters : ', sum(p.numel() for p in model.parameters() 
                    if p.requires_grad))

#Dataloader
train_dataset = MolDataset(train_keys, args.data_dir, id_to_y)
train_data_loader = DataLoader(train_dataset, args.batch_size, \
		num_workers = args.num_workers, \
		collate_fn=my_collate_fn, shuffle=True)
test_dataset = MolDataset(test_keys, args.data_dir, id_to_y)
test_data_loader = DataLoader(test_dataset, args.batch_size, \
     shuffle=False, num_workers = args.num_workers, collate_fn=my_collate_fn)

#Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, \
                             weight_decay=args.weight_decay)
loss_fn = nn.MSELoss()

#train
writer = SummaryWriter(os.path.join("runs", args.exp_name))
for epoch in range(args.num_epochs):
    st = time.time()
    tmp_st = st

    train_losses1 = []
    train_losses2 = []
    test_losses1 = []
    test_losses2 = []
    
    train_pred1 = dict()
    train_pred2 = dict()
    train_true = dict()
    
    test_pred1 = dict()
    test_pred2 = dict()
    test_true = dict()
    
    model.train()
    for i_batch, sample in enumerate(train_data_loader):
        model.zero_grad()
        if sample is None : continue
        h1, adj1, h2, adj2, A_int, dmv, dmv_rot, valid, affinity, keys = sample

        h1, adj1, h2, adj2, A_int, dmv, dmv_rot, valid, affinity = \
                h1.to(device), adj1.to(device), h2.to(device), adj2.to(device), \
                A_int.to(device), dmv.to(device), dmv_rot.to(device), \
                valid.to(device), affinity.to(device)
        pred1 = model(h1, adj1, h2, adj2, A_int, dmv, valid).sum(-1)
        pred2 = model(h1, adj1, h2, adj2, A_int, dmv_rot, valid).sum(-1)
        
        loss1 = loss_fn(pred1, affinity)
        loss2 = torch.mean(torch.max(torch.zeros_like(pred2), pred1.detach()-pred2+10)) # only consider the prediction values of rotated molecules that difference of value between two molecules are less than 10
        loss = loss1+loss2*args.loss2_ratio
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        train_losses1.append(loss1.data.cpu().numpy())
        train_losses2.append(loss2.data.cpu().numpy())
        affinity = affinity.data.cpu().numpy()
        pred1 = pred1.data.cpu().numpy()

        for i in range(len(keys)):
            train_pred1[keys[i]] = pred1[i]
            train_pred2[keys[i]] = pred2[i]
            train_true[keys[i]] = affinity[i]

        tmp_ttime = time.time()
        train_batch_time = tmp_ttime - tmp_st
        tmp_st = tmp_ttime
        print('TRAIN epoch: {}, base_loss: {:.4f}, self_supervised_loss: {:.4f}, total_loss: {:.4f}, time: {:.2f}'\
                .format(epoch, loss1, loss2, loss, train_batch_time))

    train_base_loss = np.mean(np.array(train_losses1))
    train_ss_loss = np.mean(np.array(train_losses2))
    train_total_loss = train_base_loss + train_ss_loss*args.loss2_ratio
    writer.add_scalars('train',
                       {'loss': train_base_loss,
                        'self_sup_loss': train_ss_loss,
                        'total_loss': train_total_loss},
                       epoch)

    model.eval()
    for i_batch, sample in enumerate(test_data_loader):
        model.zero_grad()
        if sample is None : continue
        h1, adj1, h2, adj2, dmv, A_int, dmv_rot, valid, affinity, keys = sample

        h1, adj1, h2, adj2, dmv, A_int, dmv_rot, valid, affinity = \
                h1.to(device), adj1.to(device), h2.to(device), adj2.to(device), \
                A_int.to(device), dmv.to(device), dmv_rot.to(device), \
                valid.to(device), affinity.to(device)
        with torch.no_grad():
            pred1 = model(h1, adj1, h2, adj2, A_int, dmv, valid).sum(-1)
            pred2 = model(h1, adj1, h2, adj2, A_int, dmv_rot, valid).sum(-1)
        
        loss1 = loss_fn(pred1, affinity)
        loss2 = torch.mean(torch.max(torch.zeros_like(pred2), 
                            pred1.detach()-pred2+10))
        loss = loss1+loss2
        test_losses1.append(loss1.data.cpu().numpy())
        test_losses2.append(loss2.data.cpu().numpy())
        affinity = affinity.data.cpu().numpy()
        pred1 = pred1.data.cpu().numpy()

        for i in range(len(keys)):
            test_pred1[keys[i]] = pred1[i]
            test_pred2[keys[i]] = pred2[i]
            test_true[keys[i]] = affinity[i]
        #if i_batch>2: break 

        tmp_etime = time.time()
        eval_batch_time = tmp_etime - tmp_st
        tmp_st = tmp_etime
        print('EVAL epoch: {}, base_loss: {:.4f}, self_supervised_loss: {:.4f}, total_loss: {:.4f}, time: {:.2f}'\
                .format(epoch, loss1, loss2, loss, eval_batch_time))

    eval_base_loss = np.mean(np.array(test_losses1))
    eval_ss_loss = np.mean(np.array(test_losses2))
    eval_total_loss = eval_base_loss + eval_ss_loss*args.loss2_ratio
    writer.add_scalars('eval',
                       {'loss': eval_base_loss,
                        'self_sup_loss': eval_ss_loss,
                        'total_loss': eval_total_loss},
                       epoch)

    #Write prediction
    if not os.path.exists("output"):
        os.mkdir("output")
    w_train = open(os.path.join("output", args.exp_name + "_" + args.train_output_filename), 'a')
    w_test = open(os.path.join("output", args.exp_name + "_" + args.test_output_filename), 'a')
    
    for k in train_pred1.keys():
        w_train.write(f'{k}\t{train_true[k]:.3f}\t{train_pred1[k]:.3f}\t{train_pred2[k]:.3f}\n')
    for k in test_pred1.keys():
        w_test.write(f'{k}\t{test_true[k]:.3f}\t{test_pred1[k]:.3f}\t{test_pred2[k]:.3f}\n')
    end = time.time()
    
    w_train.close()
    w_test.close()

    #Cal R2
    train_r2 = r2_score([train_true[k] for k in train_true.keys()], \
            [train_pred1[k] for k in train_true.keys()])
    test_r2 = r2_score([test_true[k] for k in test_true.keys()], \
            [test_pred1[k] for k in test_true.keys()])

    #Cal R 
    _, _, test_r, _, _ = \
            stats.linregress([test_true[k] for k in test_true.keys()],                        
                            [test_pred1[k] for k in test_true.keys()])
    _, _, train_r, _, _ = \
            stats.linregress([train_true[k] for k in train_true.keys()],                        
                            [train_pred1[k] for k in train_true.keys()])
    end = time.time()

    print ("epoch: {} train_losses1: {:.4f} train_losses2: {:.4f} test_losses1: {:.4f} test_losses2: {:.4f} train_r2: {:.4f} test_r2: {:.3f} time: {:.3f}"\
            .format(epoch, train_base_loss, train_ss_loss, eval_base_loss, eval_ss_loss, train_r2, test_r2, end-st))
    
    name = os.path.join(args.save_dir, args.exp_name, 'save_'+str(epoch)+'.pt')
    torch.save(model.state_dict(), name)
    
    lr = args.lr * ((args.lr_decay)**epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr             
