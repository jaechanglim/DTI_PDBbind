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
        """
        1. atom features of ligand and protein molecule goes through node embedding linear layer
        2. each atom feature vector and adjacency matrix go through graph convolution network with attention
        3. make new atom feature vector with each ligand molecule vector and protein molecule vector
        4. resulting vector go through linear layers and then sigmoid function to predict value 0 or 1
        :param h1:  atom feature one-hot vector of ligand molecule
                    shape: [# of ligand molecule's atom, property]
        :param adj1:  adjacency matrix of ligand molecule
                    shape: [# of ligand molecule's atom, # of ligand molecule's atom]
        :param h2:  atom feature one-hot vector of protein molecule
                    shape: [# of protein molecule's atom, property]
        :param adj2:  adjacency matrix of protein molecule
                    shape: [# of protein molecule's atom, # of protein molecule's atom]
        :param dmv:  distance matrix between every atoms of m1 and m2
                    shape: [# of ligand molecule's atom, # of protein molecule's atom, 3]
        :param valid: distance(sum of square of distance matrix) between each molecule's atom
                    shape: [# of ligand molecule's atom, # of protein molecule's atom, 1]
        :param DM_min:   true valid atom indices which will be used after 'my_collate_fn' makes each molecules'
                    vector and adjacency matrices into same size with zero padding and calculate property
                    shape: [# of ligand molecule's atom]
        :return:
        """
        h1 = self.node_embedding(h1) # [, n_ligand_atom, n_in_feature(dim_gnn)] 
        h2 = self.node_embedding(h2) # [, n_protein_atom, n_in_feature]
        # attention applied each molecule's property
        for i in range(len(self.gconv)):
            h1 = self.gconv[i](h1, adj1) # [, n_ligand_atom, n_out_feature(dim_gnn)]
            h2 = self.gconv[i](h2, adj2) # [, n_protein_atom, n_out_feature]

        dm = torch.sqrt(torch.pow(dmv, 2).sum(-1)+1e-10)
        adj12 = dm.clone()
        
        adj12[adj12>5] = 0
        adj12[adj12>1e-3] = 1
        adj12[adj12<1e-3] = 0
        
        dm[dm<DM_min] = 1e10
        
        # n_edge_feature = 3
        for i in range(len(self.edgeconv)):
            new_h1 = self.edgeconv[i](h1, h2, dmv, adj12) # [, n_ligand_atom, n_out_feature(dim_gnn)]
            new_h2 = self.edgeconv[i](h2, h1, \
                    dmv.permute(0,2,1,3), adj12.permute(0,2,1)) # [, n_protein_atom, n_out_feature(dim_gnn)]
            h1, h2 = new_h1, new_h2 
        
        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h = torch.cat([h1_repeat, h2_repeat], -1) # [, n_ligand_atom, n_protein_atom, 2*n_out_feature(dim_gnn)]
        
        A = self.cal_A(h).squeeze(-1)*4
        B = self.cal_B(h).squeeze(-1)*4
        C = self.cal_C(h).squeeze(-1)*4
        
        """
        morse potential
        V'(r) = A*(1-e^(-B*(r-C)))^2
        A(D_e): well depth defined relative to the dissociated atoms
        B(a): controls 'width' of the potential(the smaller a is, the larger the well)
        C(r_e): equilibrium bond distance
        """
        retval = A*(torch.pow(1-torch.exp(-B*(dm-C)),2)-1)
        """
        for i in range(retval.size(1)):
            for j in range(retval.size(2)):
                if abs(retval[0,i,j].item()) > 0.01:
                    print (i, j, retval[0,i,j].item(), dm[0,i,j].item())
        retval = retval.sum(1).sum(1)
        print (retval)
        exit(-1)
        """                
        retval = retval.sum(1).sum(1)
        return retval



























