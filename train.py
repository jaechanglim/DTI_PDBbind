import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import pickle
from sklearn.metrics import r2_score, roc_auc_score
from scipy import stats

import utils
import model 
from dataset import MolDataset, DTISampler,  tensor_collate_fn
import arguments
import sys

args = arguments.parser(sys.argv)
#Make directory for save files
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.tensorboard_dir, exist_ok=True)

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
train_data_loader = DataLoader(train_dataset,
                               args.batch_size,
                               num_workers=args.num_workers,
                               collate_fn=tensor_collate_fn,
                               shuffle=True)

test_dataset = MolDataset(test_keys, args.data_dir, id_to_y)
test_data_loader = DataLoader(test_dataset,
                              args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=tensor_collate_fn,
                              shuffle=False)

#Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, \
                             weight_decay=args.weight_decay)
loss_fn = nn.MSELoss()

#train
writer = SummaryWriter(args.tensorboard_dir)
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

        sample = utils.dic_to_device(sample, device)
        keys = sample['key']
        affinity = sample['affinity']

        pred1 = model(sample)
        pred2 = model(sample)
        
        loss1 = loss_fn(pred1.sum(-1), affinity)
        # only consider the prediction values of rotated molecules 
        #that difference of value between two molecules are less than 10
        loss2 = torch.mean(torch.max(torch.zeros_like(pred2.sum(-1)), 
                            pred1.sum(-1).detach()-pred2.sum(-1)+10))
        loss = loss1+loss2*args.loss2_ratio
        loss.backward()
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
        #print('TRAIN epoch: {}, base_loss: {:.4f}, self_supervised_loss: {:.4f}, total_loss: {:.4f}, time: {:.2f}'\
        #        .format(epoch, loss1, loss2, loss, train_batch_time))

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
        
        sample = utils.dic_to_device(sample, device)
        keys = sample['key']
        affinity = sample['affinity']

        with torch.no_grad():
            pred1 = model(sample)
            pred2 = model(sample)
        
        loss1 = loss_fn(pred1.sum(-1), affinity)
        loss2 = torch.mean(torch.max(torch.zeros_like(pred2.sum(-1)), 
                            pred1.sum(-1).detach()-pred2.sum(-1)+10))
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
        #print('EVAL epoch: {}, base_loss: {:.4f}, self_supervised_loss: {:.4f}, total_loss: {:.4f}, time: {:.2f}'\
        #        .format(epoch, loss1, loss2, loss, eval_batch_time))

    eval_base_loss = np.mean(np.array(test_losses1))
    eval_ss_loss = np.mean(np.array(test_losses2))
    eval_total_loss = eval_base_loss + eval_ss_loss*args.loss2_ratio
    writer.add_scalars('eval',
                       {'loss': eval_base_loss,
                        'self_sup_loss': eval_ss_loss,
                        'total_loss': eval_total_loss},
                       epoch)

    #Write prediction
    w_train = open(args.train_output_filename, 'w')
    w_test = open(args.eval_output_filename, 'w')
    
    for k in train_pred1.keys():
        w_train.write(f'{k}\t{train_true[k]:.3f}\t')
        w_train.write(f'{train_pred1[k].sum():.3f}\t')
        w_train.write(f'{train_pred2[k].sum():.3f}\t')
        for j in range(train_pred1[k].shape[0]):
            w_train.write(f'{train_pred1[k][j]:.3f}\t')
        w_train.write('\n')

    for k in test_pred1.keys():
        w_test.write(f'{k}\t{test_true[k]:.3f}\t')
        w_test.write(f'{test_pred1[k].sum():.3f}\t')
        w_test.write(f'{test_pred2[k].sum():.3f}\t')
        for j in range(test_pred1[k].shape[0]):
            w_test.write(f'{test_pred1[k][j]:.3f}\t')
        w_test.write('\n')

    end = time.time()
    
    w_train.close()
    w_test.close()

    #Cal R2
    train_r2 = r2_score([train_true[k] for k in train_true.keys()], \
            [train_pred1[k].sum() for k in train_true.keys()])
    test_r2 = r2_score([test_true[k] for k in test_true.keys()], \
            [test_pred1[k].sum() for k in test_true.keys()])

    #Cal R 
    _, _, test_r, _, _ = \
            stats.linregress([test_true[k] for k in test_true.keys()],                        
                            [test_pred1[k].sum() for k in test_true.keys()])
    _, _, train_r, _, _ = \
            stats.linregress([train_true[k] for k in train_true.keys()],                        
                            [train_pred1[k].sum() for k in train_true.keys()])
    end = time.time()
    if epoch==0: 
        print ("epoch\ttrain_l1\ttrain_l2\ttest_l1\ttest_l2\t"+
               "train_r2\ttest_r2\ttrain_r\ttest_r\t{time}")
    print (f"{epoch}\t{train_base_loss:.3f}\t{train_ss_loss:.3f}\t"+
            f"{eval_base_loss:.3f}\t{eval_ss_loss:.3f}\t"+
            f"{train_r2:.3f}\t{test_r2:.3f}\t"+
            f"{train_r:.3f}\t{test_r:.3f}\t{end-st:.3f}")
    
    name = os.path.join(args.save_dir, 'save_'+str(epoch)+'.pt')
    torch.save(model.state_dict(), name)
    
    lr = args.lr * ((args.lr_decay)**epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr             
