import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import GAT_gate, EdgeConv
import dataset


class DTIHarmonic(torch.nn.Module):
    def __init__(self, args):
        super(DTIHarmonic, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(54, args.dim_gnn, bias = False)

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
        self.cal_coolomb_interaction_N = nn.Sequential(
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
        self.cal_vdw_interaction_B = nn.Sequential(
                         nn.Linear(args.dim_gnn*2, 128),
                         nn.ReLU(),
                         nn.Linear(128, 1),
                         nn.Tanh()
                        )
        self.cal_vdw_interaction_N = nn.Sequential(
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
        self.delta_uff_coeff = nn.Parameter(torch.tensor([0.0]))
        self.vdw_coeff = nn.Parameter(torch.tensor([0.02]))
        self.coolomb_coeff = nn.Parameter(torch.tensor([0.0]))
        self.cal_intercept = nn.Sequential(
                             nn.Linear(args.dim_gnn*1, 128),
                             nn.ReLU(),
                             nn.Linear(128, 1),
                             #nn.Tanh()
                         )

    def cal_coolomb_interaction(self, dm, h, charge1, charge2, valid1, valid2):
        charge1_repeat = charge1.unsqueeze(2).repeat(1,1,charge2.size(1))
        charge2_repeat = charge2.unsqueeze(1).repeat(1,charge1.size(1),1)
        valid1_repeat = valid1.unsqueeze(2).repeat(1,1,valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1,valid1.size(1),1)
        A = self.cal_coolomb_interaction_A(h).squeeze(-1)
        #A = self.coolomb_coeff*self.coolomb_coeff 
        N = self.cal_coolomb_interaction_N(h).squeeze(-1)*2+1
        charge12 = charge1_repeat*charge2_repeat
        energy = A*charge12*torch.pow(1/dm, N)
        energy = energy*valid1_repeat*valid2_repeat
        energy = energy.clamp(min=-100, max=100)
        energy = energy.sum(1).sum(1).unsqueeze(-1)

        return energy  
    
    def cal_vdw_interaction(self, dm, h, vdw_radius1, vdw_radius2, 
                            vdw_epsilon, vdw_sigma, valid1, valid2):
        valid1_repeat = valid1.unsqueeze(2).repeat(1,1,valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1,valid1.size(1),1)
        #vdw_radius1_repeat = vdw_radius1.unsqueeze(2)\
        #        .repeat(1,1,vdw_radius2.size(1))
        #vdw_radius2_repeat = vdw_radius2.unsqueeze(1)\
        #        .repeat(1,vdw_radius1.size(1),1)
        #dm_0 = vdw_radius1_repeat+vdw_radius2_repeat
        B = self.cal_vdw_interaction_B(h).squeeze(-1)*0.6+0.7
        dm_0 = vdw_sigma*B
        dm_0[dm_0<0.0001] = 1
        N = self.cal_vdw_interaction_N(h).squeeze(-1)*2+5
        
        vdw1 = torch.pow(dm_0/dm, 2*N)
        vdw2 = -2*torch.pow(dm_0/dm, N)
        
        A = self.cal_vdw_interaction_A(h).squeeze(-1)*0.6+0.7
        A = A*self.vdw_coeff*self.vdw_coeff
        A = A*vdw_epsilon

        energy = A*(vdw1+vdw2)
        energy = energy*valid1_repeat*valid2_repeat
        energy = energy.clamp(max=100)
        energy = energy.sum(1).sum(1).unsqueeze(-1)

        return energy  

    def forward(self, sample, DM_min=0.5):
        h1, adj1, h2, adj2, A_int, dmv, _, _, sasa, dsasa, rotor,\
        charge1, charge2, vdw_radius1, vdw_radius2, vdw_epsilon, \
        vdw_sigma, delta_uff, valid1, valid2,\
        no_metal1, no_metal2, _ = sample.values()

        h1 = self.node_embedding(h1)  
        h2 = self.node_embedding(h2) 
        for i in range(len(self.gconv)):
            h1 = self.gconv[i](h1, adj1)
            #h2 = self.gconv[i](h2, adj2) 

        dm = torch.sqrt(torch.pow(dmv, 2).sum(-1)+1e-10)
        
        dm[dm<DM_min] = 1e10
        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) 
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) 
        h = torch.cat([h1_repeat, h2_repeat], -1) 
        
        retval = []
        for i in range(self.num_interaction_type):
            continue
            #if i not in [1,7]: continue
            
            A = self.cal_A[i](h).squeeze(-1)*4
            B = self.cal_B[i](h).squeeze(-1)*4+1
            #B = (self.cal_B[i](h).squeeze(-1)*3+1)/(4*self.sigma[i]*self.sigma[i])
            #energy = A*(B*torch.pow(dm-self.C[i],2)-1)
            energy = A*(F.tanh(B*(dm-self.C[i]))-1)*0.5
            energy = energy*A_int[:,i,:,:]
            #energy = energy.clamp(max=0.0)
            energy = energy.sum(1).sum(1).unsqueeze(-1)
            retval.append(energy)
        
        #hydrophobic contribution
        #hydrophobic = (self.sasa_coeff*sasa).unsqueeze(-1)
        #hydrophobic = hydrophobic.clamp(max=0)
        #retval.append(hydrophobic)

        #flexibility contribution
        #flexibility = (self.rotor_coeff*rotor).unsqueeze(-1)
        #flexibility = flexibility.clamp(min=0)
        #retval.append(flexibility)

        #coolomb interaction
        retval.append(self.cal_coolomb_interaction(dm, h, charge1, charge2, \
                                                   valid1, valid2))
        #vdw interaction
        retval.append(self.cal_vdw_interaction(dm, h, vdw_radius1, vdw_radius2, 
                                               vdw_epsilon, vdw_sigma,
                                               no_metal1, no_metal2))
        #delta uff
        #delta_uff = delta_uff/20
        diff_conf_energy = self.delta_uff_coeff*self.delta_uff_coeff*delta_uff
        diff_conf_energy = diff_conf_energy.unsqueeze(-1)
        retval.append(diff_conf_energy)

        #intercept
        intercept = self.cal_intercept((h1*valid1.unsqueeze(-1)).sum(1))
        #intercept = self.intercept.unsqueeze(-1).repeat(h.size(0), 1)
        retval.append(intercept)
        
        retval = torch.cat(retval, -1)
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

