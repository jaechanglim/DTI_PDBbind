import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import GAT_gate, EdgeConv

class DTIMorse(torch.nn.Module):

    def __init__(self, args):
        super(DTIMorse, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(56, args.dim_gnn, bias = False)

        self.gconv = nn.ModuleList([GAT_gate(args.dim_gnn, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        
        self.edgeconv = nn.ModuleList([EdgeConv(3, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        self.num_interaction_type = 7        
        self.cal_A = nn.ModuleList([nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) for _ in range(self.num_interaction_type)])
        
        self.cal_B = nn.ModuleList([nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) for _ in range(self.num_interaction_type)])
        
        self.cal_C = nn.ModuleList([nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) for _ in range(self.num_interaction_type)])
        
        self.A = nn.Parameter(torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]))
        self.C = nn.Parameter(torch.tensor(
                [4.639, 3.184, 4.563, 4.709, 3.356, 4.527, 3.714]))
        self.C.requires_grad = False


    def forward(self, h1, adj1, h2, adj2, A_int, dmv, valid, DM_min=0.5):
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
        
        dm[dm<DM_min] = 1e10 # to ignore too small values that makes morse potential diverge
        
        # n_edge_feature = 3
        for i in range(len(self.edgeconv)):
            new_h1 = self.edgeconv[i](h1, h2, dmv, adj12) # [, n_ligand_atom, n_out_feature(dim_gnn)]
            new_h2 = self.edgeconv[i](h2, h1, \
                    dmv.permute(0,2,1,3), adj12.permute(0,2,1)) # [, n_protein_atom, n_out_feature(dim_gnn)]
            h1, h2 = new_h1, new_h2 
        
        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h = torch.cat([h1_repeat, h2_repeat], -1) # [, n_ligand_atom, n_protein_atom, 2*n_out_feature(dim_gnn)]
        
        """
        morse potential
        V'(r) = A*(1-e^(-B*(r-C)))^2
        A(D_e): well depth defined relative to the dissociated atoms
        B(a): controls 'width' of the potential(the smaller a is, the larger the well)
        C(r_e): equilibrium bond distance
        """
        retval = 0
        #print ('type\tA\tE\tB\tdm\tC')
        for i in range(self.num_interaction_type):
            A = self.cal_A[i](h).squeeze(-1)*4
            B = self.cal_B[i](h).squeeze(-1)*0.693+0.8
            #B = 0.69314718056/1 #ln(2) = 0.69314718056
            #C = self.cal_C[i](h).squeeze(-1)*4
            #retval += self.A[i]*(torch.pow(1-torch.exp(-B*(dm-self.C[i])),2)-1)\
            #        *A_int[:,i,:,:]
            energy = A*(torch.pow(1-torch.exp(-B*(dm-self.C[i])),2)-1)\
                    *A_int[:,i,:,:]
            retval+= energy
                
        retval = retval.sum(1).sum(1)
        #print (retval[0])
        #exit(-1)
        return retval

class DTIHarmonicIS(torch.nn.Module):
    def __init__(self, args):
        super(DTIHarmonicIS, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(56, args.dim_gnn, bias=False)
        self.gconv = nn.ModuleList([GAT_gate(args.dim_gnn, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        self.edgeconv = nn.ModuleList([EdgeConv(3, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        self.num_interaction_type = 7
        self.cal_A = nn.ModuleList([nn.Sequential(
                        nn.Linear(args.dim_gnn*2, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                        nn.Sigmoid()
                       ) for _ in range(self.num_interaction_type)])

        self.cal_B = nn.ModuleList([nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) for _ in range(self.num_interaction_type)])
        
        self.cal_C = nn.ModuleList([nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) for _ in range(self.num_interaction_type)])
        
        self.C = nn.Parameter(torch.tensor(
                [4.639, 3.184, 4.563, 4.709, 3.356, 4.527, 3.714]))
        self.B_constraint = [1.159, 0.448, 0.927, 0.902, 0.349, 0.789, 0.198] 
        self.intercept = nn.Parameter(torch.tensor([0.0]))
        self.cal_intercept = nn.Sequential(
                             nn.Linear(args.dim_gnn*1, 128),
                             nn.ReLU(),
                             nn.Linear(128, 1),
                             nn.Sigmoid()
                         )

    def forward(self, h1, adj1, h2, adj2, A_int, dmv, valid, DM_min=0.5):
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

        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h = torch.cat([h1_repeat, h2_repeat], -1) # [, n_ligand_atom, n_protein_atom, 2*n_out_feature(dim_gnn)]

        """
        interaction type - average distance, std, (2*std)^2
        0. saltbridge - 4.639, 1.159, 5.373
        1. hbond - 3.184, 0.448, 0.803
        2. pication - 4.561, 0.927, 3.437
        3. pistack - 4.708, 0.902, 3.254
        4. halogen - 3.356, 0.349, 0.487
        5. waterbridge - 4.528, 0.789, 2.490
        6. hydrophobic - 3.714, 0.198, 0.157
        """

        retval = []
        for i in range(self.num_interaction_type):
            A = self.cal_A[i](h).squeeze(-1)*4
            B = self.cal_B[i](h).squeeze(-1)*(2/(3*(self.B_constraint[i]**2)))+(1/(3*(self.B_constraint[i]**2)))
            energy = A*(B*torch.pow(dm-self.C[i],2)-1)*A_int[:,i,:,:]
            retval.append(energy)
        retval = torch.stack(retval, 1).sum(2).sum(2)
        intercept = self.cal_intercept((h1*valid.unsqueeze(-1)).sum(1))*4
        retval += intercept/retval.size(-1)
        return retval


class DTIHarmonic(torch.nn.Module):
    def __init__(self, args):
        super(DTIHarmonic, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(56, args.dim_gnn, bias = False)

        self.gconv = nn.ModuleList([GAT_gate(args.dim_gnn, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        
        self.edgeconv = nn.ModuleList([EdgeConv(3, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        self.num_interaction_type = 7        
        self.cal_A = nn.ModuleList([nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) for _ in range(self.num_interaction_type)])
        
        self.cal_B = nn.ModuleList([nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) for _ in range(self.num_interaction_type)])
        
        self.cal_C = nn.ModuleList([nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        ) for _ in range(self.num_interaction_type)])
        
        self.A = nn.Parameter(torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]))
        self.C = nn.Parameter(torch.tensor(
                [4.639, 3.184, 4.563, 4.709, 3.356, 4.527, 3.714]))
        #self.C.requires_grad = False
        self.intercept = nn.Parameter(torch.tensor([0.0]))
        self.cal_intercept = nn.Sequential(
                             nn.Linear(args.dim_gnn*1, 128),
                             nn.ReLU(),
                             nn.Linear(128, 1),
                             nn.Sigmoid()
                         )
        
    def forward(self, h1, adj1, h2, adj2, A_int, dmv, valid, DM_min=0.5):
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
        """
        # n_edge_feature = 3
        for i in range(len(self.edgeconv)):
            new_h1 = self.edgeconv[i](h1, h2, dmv, adj12) # [, n_ligand_atom, n_out_feature(dim_gnn)]
            new_h2 = self.edgeconv[i](h2, h1, \
                    dmv.permute(0,2,1,3), adj12.permute(0,2,1)) # [, n_protein_atom, n_out_feature(dim_gnn)]
            h1, h2 = new_h1, new_h2 
        """ 
        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h = torch.cat([h1_repeat, h2_repeat], -1) # [, n_ligand_atom, n_protein_atom, 2*n_out_feature(dim_gnn)]
        
        """
        morse potential
        V'(r) = A*(1-e^(-B*(r-C)))^2
        A(D_e): well depth defined relative to the dissociated atoms
        B(a): controls 'width' of the potential(the smaller a is, the larger the well)
        C(r_e): equilibrium bond distance
        """
        retval = []
        #A_int[:,6,:,:] = 0.0
        for i in range(self.num_interaction_type):
            A = self.cal_A[i](h).squeeze(-1)*4
            #if i==1: A+=1
            B = self.cal_B[i](h).squeeze(-1)*3+1
            energy = A*(B*torch.pow(dm-self.C[i],2)-1)*A_int[:,i,:,:]
            retval.append(energy)
            """ 
            if i==0: print ('type\tA\tE\tB\tdm\tC')
            for j in range(A_int.size(2)):
                for k in range(A_int.size(3)):
                    if A_int[0,i,j,k]>0:
                        print (f'{i}\t{A[0,j,k]:.3f}\t{energy[0,j,k]:.3f}\t{B[0,j,k]:.3f}\t{dm[0,j,k]:.3f}\t{self.C[i]:.3f}')
            print ('energy: ', energy.sum())    
            """ 
        retval = torch.stack(retval, 1).sum(2).sum(2)
        #retval += self.intercept/retval.size(-1)
        intercept = self.cal_intercept((h1*valid.unsqueeze(-1)).sum(1))*4
        retval += intercept/retval.size(-1)
        return retval

class DTIMorseAllPair(torch.nn.Module):

    def __init__(self, args):
        super(DTIMorseAllPair, self).__init__()
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
        

    def forward(self, h1, adj1, h2, adj2, A_int, dmv, valid, DM_min=0.5):
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
        """ 
        # n_edge_feature = 3
        for i in range(len(self.edgeconv)):
            new_h1 = self.edgeconv[i](h1, h2, dmv, adj12) # [, n_ligand_atom, n_out_feature(dim_gnn)]
            new_h2 = self.edgeconv[i](h2, h1, \
                    dmv.permute(0,2,1,3), adj12.permute(0,2,1)) # [, n_protein_atom, n_out_feature(dim_gnn)]
            h1, h2 = new_h1, new_h2 
        """
        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h = torch.cat([h1_repeat, h2_repeat], -1) # [, n_ligand_atom, n_protein_atom, 2*n_out_feature(dim_gnn)]
        
        A = self.cal_A(h).squeeze(-1)*2
        B = self.cal_B(h).squeeze(-1)*2
        C = self.cal_C(h).squeeze(-1)*3+2.5
        
        """
        morse potential
        V'(r) = A*(1-e^(-B*(r-C)))^2
        A(D_e): well depth defined relative to the dissociated atoms
        B(a): controls 'width' of the potential(the smaller a is, the larger the well)
        C(r_e): equilibrium bond distance
        """
        retval = A*(torch.pow(1-torch.exp(-B*(dm-C)),2)-1)
        retval = retval.sum(1).sum(1).unsqueeze(-1)
        return retval

