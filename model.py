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
        if args.edgeconv: 
            num_filter = int(10.0/args.filter_spacing)+1 
            self.filter_center = torch.Tensor([args.filter_spacing*i for i 
                    in range(num_filter)])
            self.filter_gamma = args.filter_gamma
            self.edgeconv = nn.ModuleList([EdgeConv(num_filter, args.dim_gnn) \
                                        for _ in range(args.n_gnn)])
        self.num_interaction_type = len(dataset.interaction_types)
        
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
        self.vina_hbond_coeff = nn.Parameter(torch.tensor([1.0])) 
        self.vina_hydrophobic_coeff = nn.Parameter(torch.tensor([1.0])) 
        self.vdw_coeff = nn.Parameter(torch.tensor([1.0])) 
        self.torsion_coeff = nn.Parameter(torch.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([1.0]))
        self.intercept = nn.Parameter(torch.tensor([0.0]))

    def cal_intercept(self, h, valid1, valid2, dm):
        valid1_repeat = valid1.unsqueeze(2).repeat(1,1,valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1,valid1.size(1),1)
        C1 = self.cal_intercept_A(h).squeeze(-1)*0.01
        C2 = self.cal_intercept_B(h).squeeze(-1)*0.1+0.1
        retval = C1*torch.exp(-torch.pow(C2*dm, 2))
        retval = retval*valid1_repeat*valid2_repeat
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

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

    def vina_steric(self, dm, h, vdw_radius1, vdw_radius2, valid1, valid2):
        valid1_repeat = valid1.unsqueeze(2).repeat(1,1,valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1,valid1.size(1),1)
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2)\
                .repeat(1,1,vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1)\
                .repeat(1,vdw_radius1.size(1),1)
        dm_0 = vdw_radius1_repeat+vdw_radius2_repeat
        dm = dm-dm_0
        g1 = torch.exp(-torch.pow(dm/0.5,2))*-0.0356  
        g2 = torch.exp(-torch.pow((dm-3)/2,2))*-0.00516
        repulsion = dm*dm*0.84
        zero_vec = torch.zeros_like(repulsion)
        repulsion = torch.where(dm > 0, zero_vec, repulsion)
        retval = g1+g2+repulsion
        retval = retval*valid1_repeat*valid2_repeat
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval
    
    def vina_hbond(self, dm, h, vdw_radius1, vdw_radius2, A):
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2)\
                .repeat(1,1,vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1)\
                .repeat(1,vdw_radius1.size(1),1)
        B = self.cal_vdw_interaction_B(h).squeeze(-1)*0.2
        dm_0 = vdw_radius1_repeat+vdw_radius2_repeat+B
        dm = dm-dm_0
        retval = dm*A/-0.7
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval*-self.vina_hbond_coeff*self.vina_hbond_coeff
        #retval = retval.clamp(min=0.0, max=1.0)*-0.587
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval
    
    def vina_hydrophobic(self, dm, h, vdw_radius1, vdw_radius2, A):
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2)\
                .repeat(1,1,vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1)\
                .repeat(1,vdw_radius1.size(1),1)
        B = self.cal_vdw_interaction_B(h).squeeze(-1)*0.2
        dm_0 = vdw_radius1_repeat+vdw_radius2_repeat+B
        dm = dm-dm_0

        retval = (-dm+1.5)*A
        retval = retval.clamp(min=0.0, max=1.0)
        #retval = retval.clamp(min=0.0, max=1.0)*-0.0351
        retval = retval*-self.vina_hydrophobic_coeff*self.vina_hydrophobic_coeff
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval
 
    def cal_vdw_interaction(self, dm, h, vdw_radius1, vdw_radius2, 
                            vdw_epsilon, vdw_sigma, valid1, valid2):
        valid1_repeat = valid1.unsqueeze(2).repeat(1,1,valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1,valid1.size(1),1)
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2)\
                .repeat(1,1,vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1)\
                .repeat(1,vdw_radius1.size(1),1)

        B = self.cal_vdw_interaction_B(h).squeeze(-1)*0.2
        dm_0 = vdw_radius1_repeat+vdw_radius2_repeat+B
        #dm_0 = vdw_sigma
        dm_0[dm_0<0.0001] = 1
        N = self.cal_vdw_interaction_N(h).squeeze(-1)+5.5
        
        vdw1 = torch.pow(dm_0/dm, 2*N)
        vdw2 = -2*torch.pow(dm_0/dm, N)

        A = self.cal_vdw_interaction_A(h).squeeze(-1)*0.0356
        #A = A*self.vdw_coeff*self.vdw_coeff
        #A = A*vdw_epsilon

        energy = vdw1+vdw2
        energy = energy.clamp(max=100)
        energy = energy*valid1_repeat*valid2_repeat
        energy = A*energy
        energy = energy.sum(1).sum(1).unsqueeze(-1)
        return energy  

    def cal_torsion_energy(self, torsion_energy):
        retval=torsion_energy*self.vdw_coeff*self.vdw_coeff
        #retval=torsion_energy*self.torsion_coeff*self.torsion_coeff
        return retval.unsqueeze(-1)

    def cal_distance_matrix(self, p1, p2, dm_min):
        p1_repeat = p1.unsqueeze(2).repeat(1,1,p2.size(1),1)
        p2_repeat = p2.unsqueeze(1).repeat(1,p1.size(1),1,1)
        dm = torch.sqrt(torch.pow(p1_repeat-p2_repeat, 2).sum(-1)+1e-10)
        replace_vec = torch.ones_like(dm)*1e10
        dm = torch.where(dm<dm_min, replace_vec, dm)
        return dm
    
    def forward(self, sample, DM_min=0.5, cal_der_loss=False):
        h1, adj1, h2, adj2, A_int, dmv, _, pos1, pos2, _, sasa, dsasa, rotor,\
        charge1, charge2, vdw_radius1, vdw_radius2, vdw_epsilon, \
        vdw_sigma, delta_uff, valid1, valid2,\
        no_metal1, no_metal2, _ = sample.values()

        h1 = self.node_embedding(h1)  
        h2 = self.node_embedding(h2) 
        
        for i in range(len(self.gconv)):
            h1 = self.gconv[i](h1, adj1)
            h2 = self.gconv[i](h2, adj2) 
        
        pos1.requires_grad=True
        dm = self.cal_distance_matrix(pos1, pos2, DM_min)
        if self.args.edgeconv:
            edge = dm.unsqueeze(-1).repeat(1,1,1,self.filter_center.size(-1))
            filter_center = self.filter_center.unsqueeze(0).\
                            unsqueeze(0).unsqueeze(0).to(h1.device)

            edge = torch.exp(-torch.pow(edge-filter_center,2)*self.filter_gamma)
            edge = edge.detach()
            adj12 = dm.clone().detach()

            adj12[adj12>5] = 0
            adj12[adj12>1e-3] = 1
            adj12[adj12<1e-3] = 0
            
            for i in range(len(self.edgeconv)):
                new_h1 = self.edgeconv[i](h1, h2, edge, adj12) # [, n_ligand_atom, n_out_feature(dim_gnn)]
                new_h2 = self.edgeconv[i](h2, h1, \
                        edge.permute(0,2,1,3), adj12.permute(0,2,1)) # [, n_protein_atom, n_out_feature(dim_gnn)]
                h1, h2 = new_h1, new_h2

        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) 
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) 
        h = torch.cat([h1_repeat, h2_repeat], -1) 
        
        retval = []
        
        #coolomb interaction
        #retval.append(self.cal_coolomb_interaction(dm, h, charge1, charge2, \
        #                                           valid1, valid2))
        #vdw interaction
        retval.append(self.cal_vdw_interaction(dm, h, vdw_radius1, vdw_radius2, 
                                               vdw_epsilon, vdw_sigma,
                                               no_metal1, no_metal2))

        retval.append(self.vina_hbond(dm, h, vdw_radius1, vdw_radius2, A_int[:,1]))
        retval.append(self.vina_hbond(dm, h, vdw_radius1, vdw_radius2, A_int[:,-1]))
        retval.append(self.vina_hydrophobic(dm, h, vdw_radius1, vdw_radius2, 
            A_int[:,-2]))
        retval.append(self.cal_torsion_energy(delta_uff))
        #intercept        
        #intercept = torch.stack([self.intercept 
        #                        for _ in range(retval[0].size(0))])
        #retval.append(intercept)
        retval = torch.cat(retval, -1)
        retval = retval/(1+self.rotor_coeff*self.rotor_coeff*rotor.unsqueeze(-1))
        #retval.sum().backward(retain_graph=True)
        #minimum_loss = torch.pow(dm.grad.sum(1).sum(1),2).mean()
        if cal_der_loss:
            minimum_loss = torch.autograd.grad(retval.sum(), pos1, 
                    retain_graph=True, create_graph=True)[0]
            minimum_loss2 = torch.pow(minimum_loss.sum(1), 2).mean()
            minimum_loss3 = torch.autograd.grad(minimum_loss.sum(), pos1,
                    retain_graph=True, create_graph=True)[0]                                    
            minimum_loss3 = -minimum_loss3.sum(1).sum(1).mean()    
        else:
            minimum_loss2 = torch.zeros_like(retval).sum()
            minimum_loss3 = torch.zeros_like(retval).sum()
        return retval, minimum_loss2, minimum_loss3
