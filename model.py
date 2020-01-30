import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import GAT_gate, EdgeConv
import dataset


class DTIMorse(torch.nn.Module):

    def __init__(self, args):
        super(DTIMorse, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(56, args.dim_gnn, bias = False)

        self.gconv = nn.ModuleList([GAT_gate(args.dim_gnn, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        
        self.edgeconv = nn.ModuleList([EdgeConv(3, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        self.num_interaction_type = len(dataset.interaction_types)
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

class DTIHarmonic(torch.nn.Module):
    def __init__(self, args):
        super(DTIHarmonic, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(56, args.dim_gnn, bias = False)

        self.gconv = nn.ModuleList([GAT_gate(args.dim_gnn, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        
        self.edgeconv = nn.ModuleList([EdgeConv(3, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        self.num_interaction_type = len(dataset.interaction_types)
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
        
        self.cal_coolomb_interaction_A = nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        )
        self.cal_vdw_interaction_A = nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        )
        self.coolomb_distance = nn.Parameter(torch.tensor([3.0])) 
        
        self.A = nn.Parameter(torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]))
        self.C = nn.Parameter(torch.tensor(
                [4.639, 3.184, 4.563, 4.709, 3.356, 4.527, 3.714, 2.313]))
        self.sigma = [1.159, 0.448, 0.927, 0.902, 0.349, 0.789, 0.198, 0.317]
        
        self.intercept = nn.Parameter(torch.tensor([0.0]))
        self.intercept.requires_grad=False
        self.sasa_coeff = nn.Parameter(torch.tensor([0.0]))
        self.dsasa_coeff = nn.Parameter(torch.tensor([0.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.0]))
        self.vdw_coeff = nn.Parameter(torch.tensor([0.0]))
        self.coolomb_coeff = nn.Parameter(torch.tensor([0.0]))
        self.cal_intercept = nn.Sequential(
                             nn.Linear(args.dim_gnn*1, 128),
                             nn.ReLU(),
                             nn.Linear(128, 1),
                             nn.Sigmoid()
                         )

    def cal_coolomb_interaction(self, dm, h, charge1, charge2, valid1, valid2):
        charge1_repeat = charge1.unsqueeze(2).repeat(1,1,charge2.size(1))
        charge2_repeat = charge2.unsqueeze(1).repeat(1,charge1.size(1),1)
        valid1_repeat = valid1.unsqueeze(2).repeat(1,1,valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1,valid1.size(1),1)
        #A = self.cal_coolomb_interaction_A(h).squeeze(-1)*4
        A = self.coolomb_coeff 
        energy = A*charge1_repeat*charge2_repeat/dm
        energy = energy*valid1_repeat*valid2_repeat
        energy = energy.clamp(min=-100, max=100)
        energy = energy.sum(1).sum(1).unsqueeze(-1)

        return energy  
    
    def cal_vdw_interaction(self, dm, h, vdw_radius1, vdw_radius2, 
                            valid1, valid2):
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2)\
                .repeat(1,1,vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1)\
                .repeat(1,vdw_radius1.size(1),1)
        valid1_repeat = valid1.unsqueeze(2).repeat(1,1,valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1,valid1.size(1),1)
        #A = self.cal_vdw_interaction_A(h).squeeze(-1)*1.5
        
        dm_0 = vdw_radius1_repeat+vdw_radius2_repeat
        
        vdw1 = torch.pow(dm_0/dm, 8)
        vdw2 = -2*torch.pow(dm_0/dm, 4)
        energy = self.vdw_coeff*(vdw1+vdw2)*valid1_repeat*valid2_repeat
        energy = energy.clamp(max=100)
        energy = energy.sum(1).sum(1).unsqueeze(-1)

        return energy  

    def forward(self, h1, adj1, h2, adj2, A_int, dmv, sasa, dsasa, rotor, 
                charge1, charge2, vdw_radius1, vdw_radius2, 
                valid1, valid2, no_metal1, no_metal2, DM_min=0.5):
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
        
        dm[dm<DM_min] = 1e10
        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h = torch.cat([h1_repeat, h2_repeat], -1) # [, n_ligand_atom, n_protein_atom, 2*n_out_feature(dim_gnn)]
        
        retval = []
        #A_int[:,6,:,:] = 0.0
        for i in range(self.num_interaction_type):
            A = self.cal_A[i](h).squeeze(-1)*4
            if i==1: A=A+1
            #B = (self.cal_B[i](h).squeeze(-1)+1)*self.sigma[i]
            B = (self.cal_B[i](h).squeeze(-1)*3+1)/(4*self.sigma[i]*self.sigma[i])
            energy = A*(B*torch.pow(dm-self.C[i],2)-1)
            energy = energy*A_int[:,i,:,:]
            energy = energy.clamp(max=0.0)
            energy = energy.sum(1).sum(1).unsqueeze(-1)
            retval.append(energy)
        
        #hydrophobic contribution
        hydrophobic = (self.sasa_coeff*sasa).unsqueeze(-1)
        hydrophobic = hydrophobic.clamp(max=0)
        retval.append(hydrophobic)
        
        #flexibility contribution
        flexibility = (self.rotor_coeff*rotor).unsqueeze(-1)
        flexibility = flexibility.clamp(min=0)
        retval.append(flexibility)

        #coolomb interaction
        retval.append(self.cal_coolomb_interaction(dm, h, charge1, charge2, \
                                                   valid1, valid2))
        #vdw interaction
        retval.append(self.cal_vdw_interaction(dm, h, vdw_radius1, vdw_radius2, \
                                                   no_metal1, no_metal2))

        #intercept
        intercept = self.intercept.unsqueeze(-1).repeat(h.size(0), 1)
        retval.append(intercept)
        
        retval = torch.cat(retval, -1)
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


class _DTIHarmonic(torch.nn.Module):
    def __init__(self, args):
        super(_DTIHarmonic, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(56, args.dim_gnn, bias = False)

        self.gconv = nn.ModuleList([GAT_gate(args.dim_gnn, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        
        self.edgeconv = nn.ModuleList([EdgeConv(3, args.dim_gnn) \
                                    for _ in range(args.n_gnn)])
        self.num_interaction_type = len(dataset.interaction_types)
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
        
        self.cal_coolomb_interaction_A = nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        )
        self.cal_vdw_interaction_A = nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Sigmoid()
                        )
        self.coolomb_distance = nn.Parameter(torch.tensor([3.0])) 
        
        self.A = nn.Parameter(torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]))
        self.C = nn.Parameter(torch.tensor(
                [4.639, 3.184, 4.563, 4.709, 3.356, 4.527, 3.714, 2.313]))
        self.sigma = [1.159, 0.448, 0.927, 0.902, 0.349, 0.789, 0.198, 0.317]
        
        self.intercept = nn.Parameter(torch.tensor([0.0]))
        self.intercept.requires_grad=False
        self.sasa_coeff = nn.Parameter(torch.tensor([0.0]))
        self.dsasa_coeff = nn.Parameter(torch.tensor([0.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.0]))
        self.vdw_coeff = nn.Parameter(torch.tensor([0.0]))
        self.coolomb_coeff = nn.Parameter(torch.tensor([0.0]))
        self.cal_intercept = nn.Sequential(
                             nn.Linear(args.dim_gnn*1, 128),
                             nn.ReLU(),
                             nn.Linear(128, 1),
                             nn.Sigmoid()
                         )

    def cal_coolomb_interaction(self, dm, h, charge1, charge2, valid1, valid2):
        charge1_repeat = charge1.unsqueeze(2).repeat(1,1,charge2.size(1))
        charge2_repeat = charge2.unsqueeze(1).repeat(1,charge1.size(1),1)
        valid1_repeat = valid1.unsqueeze(2).repeat(1,1,valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1,valid1.size(1),1)
        #A = self.cal_coolomb_interaction_A(h).squeeze(-1)*4
        A = self.coolomb_coeff 
        energy = A*charge1_repeat*charge2_repeat/dm
        energy = energy*valid1_repeat*valid2_repeat
        energy = energy.clamp(min=-100, max=100)
        energy = energy.sum(1).sum(1).unsqueeze(-1)

        return energy  
    
    def cal_vdw_interaction(self, dm, h, vdw_radius1, vdw_radius2, 
                            valid1, valid2):
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2)\
                .repeat(1,1,vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1)\
                .repeat(1,vdw_radius1.size(1),1)
        valid1_repeat = valid1.unsqueeze(2).repeat(1,1,valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1,valid1.size(1),1)
        #A = self.cal_vdw_interaction_A(h).squeeze(-1)*1.5
        
        dm_0 = vdw_radius1_repeat+vdw_radius2_repeat
        
        vdw1 = torch.pow(dm_0/dm, 8)
        vdw2 = -2*torch.pow(dm_0/dm, 4)
        energy = self.vdw_coeff*(vdw1+vdw2)*valid1_repeat*valid2_repeat
        energy = energy.clamp(max=100)
        energy = energy.sum(1).sum(1).unsqueeze(-1)

        return energy  

    def forward(self, dic, DM_min=0.5):
        h1, adj1, h2, adj2, A_int, dmv, _, __, sasa, dsasa, rotor,\
        charge1, charge2, vdw_radius1, vdw_radius2, valid1, valid2,\
        no_metal1, no_metal2, ___ = dic.values()
        
        h1 = self.node_embedding(h1) # [, n_ligand_atom, n_in_feature(dim_gnn)] 
        h2 = self.node_embedding(h2) # [, n_protein_atom, n_in_feature]
        # attention applied each molecule's property
        for i in range(len(self.gconv)):
            h1 = self.gconv[i](h1, adj1) # [, n_ligand_atom, n_out_feature(dim_gnn)]
            h2 = self.gconv[i](h2, adj2) # [, n_protein_atom, n_out_feature]

        dm = torch.sqrt(torch.pow(dmv, 2).sum(-1)+1e-10)
        
        dm[dm<DM_min] = 1e10
        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) # [, n_ligand_atom, n_protein_atom, n_out_feature(dim_gnn)]
        h = torch.cat([h1_repeat, h2_repeat], -1) # [, n_ligand_atom, n_protein_atom, 2*n_out_feature(dim_gnn)]
        
        retval = []
        #A_int[:,6,:,:] = 0.0
        for i in range(self.num_interaction_type):
            A = self.cal_A[i](h).squeeze(-1)*4
            if i==1: A=A+1
            #B = (self.cal_B[i](h).squeeze(-1)+1)*self.sigma[i]
            B = (self.cal_B[i](h).squeeze(-1)*3+1)/(4*self.sigma[i]*self.sigma[i])
            energy = A*(B*torch.pow(dm-self.C[i],2)-1)
            energy = energy*A_int[:,i,:,:]
            energy = energy.clamp(max=0.0)
            energy = energy.sum(1).sum(1).unsqueeze(-1)
            retval.append(energy)
        
        #hydrophobic contribution
        hydrophobic = (self.sasa_coeff*sasa).unsqueeze(-1)
        hydrophobic = hydrophobic.clamp(max=0)
        retval.append(hydrophobic)
        
        #flexibility contribution
        flexibility = (self.rotor_coeff*rotor).unsqueeze(-1)
        flexibility = flexibility.clamp(min=0)
        retval.append(flexibility)

        #coolomb interaction
        retval.append(self.cal_coolomb_interaction(dm, h, charge1, charge2, \
                                                   valid1, valid2))
        #vdw interaction
        retval.append(self.cal_vdw_interaction(dm, h, vdw_radius1, vdw_radius2, \
                                                   no_metal1, no_metal2))

        #intercept
        intercept = self.intercept.unsqueeze(-1).repeat(h.size(0), 1)
        retval.append(intercept)
        
        retval = torch.cat(retval, -1)
        return retval
