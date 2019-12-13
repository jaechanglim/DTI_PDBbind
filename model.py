import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import GAT_gate, EdgeConv

class DTIPredictor(torch.nn.Module):

    def __init__(self, args):
        super(DTIPredictor, self).__init__()
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
        

    def forward(self, H1, A1, H2, A2, DM, V, DM_min=0.5):

        H1 = self.node_embedding(H1) 
        H2 = self.node_embedding(H2) 
        for i in range(len(self.gconv)):
            H1 = self.gconv[i](H1, A1)
            H2 = self.gconv[i](H2, A2)


        VDM2 = torch.sqrt(torch.pow(DM, 2).sum(-1)+1e-9)
        VDM1 = VDM2.clone()
        
        VDM1[VDM1>5] = 0
        VDM1[VDM1>1e-3] = 1
        VDM1[VDM1<1e-3] = 0
        
        VDM2[VDM2<DM_min] = 1e10
        
        for i in range(len(self.edgeconv)):
            new_H1 = self.edgeconv[i](H1, H2, DM, VDM1 )
            new_H2 = self.edgeconv[i](H2, H1, \
                    DM.permute(0,2,1,3), VDM1.permute(0,2,1))
            H1, H2 = new_H1, new_H2 
        
        H1_repeat = H1.unsqueeze(2).repeat(1, 1, H2.size(1), 1)
        H2_repeat = H2.unsqueeze(1).repeat(1, H1.size(1), 1, 1)
        H = torch.cat([H1_repeat, H2_repeat], -1)
        
        A = self.cal_A(H).squeeze(-1)*2
        B = self.cal_B(H).squeeze(-1)*2
        C = self.cal_C(H).squeeze(-1)*4+1
        
        retval = A*(torch.pow(1-torch.exp(-B*(VDM2-C)),2)-1)
        retval = retval.sum(1).sum(1)
        return retval



























