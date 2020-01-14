import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import GAT_gate, EdgeConv

class DTILJPredictor(torch.nn.Module):

    def __init__(self, args):
        super(DTILJPredictor, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(56, args.dim_gnn, bias = False)

        self.gconv = nn.ModuleList([GAT_gate(args.dim_gnn, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        
        self.edgeconv = nn.ModuleList([EdgeConv(3, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        
        self.cal_A = nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) 
        
        self.cal_B = nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) 
        
        self.cal_C = nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) 
        

    def forward(self, h1, adj1, h2, adj2, dmv, valid, DM_min=0.5):

        h1 = self.node_embedding(h1) 
        h2 = self.node_embedding(h2) 
        for i in range(len(self.gconv)):
            h1 = self.gconv[i](h1, adj1)
            h2 = self.gconv[i](h2, adj2)

        dm = torch.sqrt(torch.pow(dmv, 2).sum(-1)+1e-10)
        adj12 = dm.clone()
        
        adj12[adj12>5] = 0
        adj12[adj12>1e-3] = 1
        adj12[adj12<1e-3] = 0
        
        dm[dm<DM_min] = 1e10 # to ignore too small values that makes morse potential diverge
        
        for i in range(len(self.edgeconv)):
            new_h1 = self.edgeconv[i](h1, h2, dmv, adj12 )
            new_h2 = self.edgeconv[i](h2, h1, \
                    dmv.permute(0,2,1,3), adj12.permute(0,2,1))
            h1, h2 = new_h1, new_h2 
        
        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1)
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1)
        h = torch.cat([h1_repeat, h2_repeat], -1)
        
        A = self.cal_A(h).squeeze(-1)*2
        B = self.cal_B(h).squeeze(-1)*2
        C = self.cal_C(h).squeeze(-1)*4+1
        
        retval = A*(torch.pow(1-torch.exp(-B*(dm-C)),2)-1)
        retval = retval.sum(1).sum(1)
        return retval



























