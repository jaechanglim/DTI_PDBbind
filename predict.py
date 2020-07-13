import utils
import numpy as np
import model 
import os
import torch
import time
import sys
import arguments
import dataset
from rdkit.Chem.rdmolops import GetAdjacencyMatrix, GetDistanceMatrix
from rdkit import Chem

#print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
def write(of, model, pred, time, args, extra_data=None):
    with open(f'{of}', 'w') as w:
        w.write('#Parameter\n')
        w.write(f'Local opt: {args.local_opt}\n')
        w.write(f'Hbond coeff: {model.vina_hbond_coeff.data.cpu().numpy()[0]:.3f}\n')
        w.write(f'Hydrophobic coeff: {model.vina_hydrophobic_coeff.data.cpu().numpy()[0]:.3f}\n')
        w.write(f'Rotor coeff: {model.rotor_coeff.data.cpu().numpy()[0]:.3f}\n')
        w.write('\n')
        
        if extra_data is not None:
            w.write('#Extra data\n')
            for k in extra_data.keys():
                w.write(f'{k}: {extra_data[k]}\n')
            w.write('\n')

        w.write('#Prediction\n')
        w.write(f'Total prediction: {pred.sum():.3f} kcal/mol\n')
        w.write(f'VDW : {pred[0]:.3f} kcal/mol\n')
        w.write(f'Hbond : {pred[1]:.3f} kcal/mol\n')
        w.write(f'Metal : {pred[2]:.3f} kcal/mol\n')
        w.write(f'Hydrophobic : {pred[3]:.3f} kcal/mol\n')
        w.write(f'\nTime : {time} s\n')
    return

<<<<<<< HEAD

=======
def cal_vdw_energy(dm, dm_0, vdw_A, vdw_N, is_last=False):
    vdw1 = torch.pow(dm_0/dm, 2*vdw_N)
    vdw2 = -2*torch.pow(dm_0/dm, vdw_N)
    energy = vdw1+vdw2
    energy = energy.clamp(max=100)
    energy = vdw_A*energy
    if is_last:
        return energy.sum(-1)[0]
    energy = energy.sum()
    return energy

def cal_hbond_energy(dm, dm_0, coeff, A, is_last=False):
    eff_dm = dm-dm_0
    energy = eff_dm*A/-0.7
    energy = energy.clamp(min=0.0, max=1.0)

    pair = energy.detach()
    pair[pair>0] = 1
    n_ligand_hbond = pair.sum(2)
    n_ligand_hbond[n_ligand_hbond<0.001] = 1

    energy = energy/(n_ligand_hbond.unsqueeze(-1))
    energy = energy*-coeff
    if is_last:
        return energy.sum(-1)[0]
    energy = energy.sum()
    return energy

def cal_hydrophobic_energy(dm, dm_0, coeff, A, is_last=False):
    eff_dm = dm-dm_0
    energy = (-eff_dm+1.5)*A
    energy = energy.clamp(min=0.0, max=1.0)
    energy = energy*-coeff
    if is_last: return energy.sum(-1)[0]
    energy = energy.sum()
    return energy
>>>>>>> 51954eaec940f7aee4841d882309a0f340a6585f

def cal_internal_vdw_energy(dm, topological_dm, epsilon, sigma, is_last=False):
    dm = dm.squeeze(0)
    energy1 = torch.pow(sigma/dm, 12)
    energy2 = -2*torch.pow(sigma/dm, 6)
    energy = epsilon*(energy1+energy2)
    energy[topological_dm<4] = 0.0
    if is_last:
        return energy.sum(-1)[0]
    energy = energy.sum()
    return energy

def make_ring_matrix(m):
    ssr = Chem.GetSymmSSSR(m)
    natoms = m.GetNumAtoms()
    retval = np.zeros((natoms, natoms))
    for indice in ssr:
        for i1 in indice:
            for i2 in indice:
                retval[i1,i2] = 1
    #print (retval)
    return retval                

def make_conjugate_matrix(m):
    from rdkit.Chem.rdchem import ResonanceMolSupplier
    suppl = ResonanceMolSupplier(m)
    natoms = m.GetNumAtoms()
    retval = np.zeros((natoms, natoms))
    groups = np.zeros((natoms,))
    for i in range(natoms):
        groups[i]=suppl.GetAtomConjGrpIdx(i)
    for i in range(natoms):        
        for j in range(natoms):        
            if groups[i]==groups[j] and groups[i]<natoms:
                retval[i,j]=1

    return retval                


def distance_fix_pair(m):
    #adjacency matrix
    adj = GetAdjacencyMatrix(m).astype(float)
    adj += np.eye(len(adj)).astype(float)
    adj_sec_neighbor = np.matmul(adj, adj)
    adj += make_ring_matrix(m).astype(float)
    adj += make_conjugate_matrix(m).astype(float)
    #adj[adj>1.0] = 1.0
    adj = np.matmul(adj, adj)
    adj+=adj_sec_neighbor
    adj[adj>1] = 1
    return adj

def write_molecule(filename, m, pos):
    if pos is not None:
        c = m.GetConformers()[0]
        for i in range(m.GetNumAtoms()):  
            c.SetAtomPosition(i,pos[i].tolist())
    if filename[-4:]=='.sdf':
        w = Chem.SDWriter(filename)
        w.write(m)
        w.close()
    return

def local_optimize(model, lf, pf, of, loof, args, device):
    st = time.time()

    #read ligand and protein. Then, convert to rdkit object
    m1 = utils.read_molecule(lf)
    m2 = utils.extract_binding_pocket(m1, pf)

    #preprocess: convert rdkit mol obj to feature
    sample = dataset.mol_to_feature(m1, m1, m2, None, 0.0)
    sample['affinity'] = 0.0
    sample['key'] = 'None'
    sample = dataset.tensor_collate_fn([sample])
    sample = utils.dic_to_device(sample, device)
    
    with torch.no_grad():
        #get embedding vector
        h1, h2 = model.get_embedding_vector(sample)
        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1) 
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1) 
        h = torch.cat([h1_repeat, h2_repeat], -1) 

        #vdw radius parameter
        dev_vdw_radius = model.cal_vdw_interaction_B(h).squeeze(-1)
        dev_vdw_radius = dev_vdw_radius*args.dev_vdw_radius
        vdw_radius1, vdw_radius2 = sample['vdw_radius1'], sample['vdw_radius2']
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2)\
                .repeat(1,1,vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1)\
                .repeat(1,vdw_radius1.size(1),1)
        sum_vdw_radius = vdw_radius1_repeat+vdw_radius2_repeat+dev_vdw_radius
        
        #vdw interaction
        vdw_N = args.vdw_N
        vdw_A = model.cal_vdw_interaction_A(h).squeeze(-1)
        vdw_A = vdw_A*(args.max_vdw_interaction-args.min_vdw_interaction)
        vdw_A = vdw_A + args.min_vdw_interaction

        #hbond and hydrophobic
        hbond_coeff = model.vina_hbond_coeff*model.vina_hbond_coeff
        hydrophobic_coeff = model.vina_hydrophobic_coeff*model.vina_hydrophobic_coeff
        
        pos1, pos2, A_int = sample['pos1'], sample['pos2'], sample['A_int']
        epsilon, sigma = dataset.get_epsilon_sigma(m1,m1,False)
        epsilon = torch.from_numpy(epsilon)
        sigma = torch.from_numpy(sigma)

        fix_pair = torch.from_numpy(distance_fix_pair(m1))
        initial_dm_internal = model.cal_distance_matrix(pos1, pos1, 0.5)
        topological_dm=torch.from_numpy(GetDistanceMatrix(m1))


    #optimizer
    pos1.requires_grad = True
    optimizer = torch.optim.Adam([pos1], lr=0.01)
    
    for iter in range(100):
        optimizer.zero_grad()
        dm = model.cal_distance_matrix(pos1, pos2, 0.5)
        dm_internal = model.cal_distance_matrix(pos1, pos1, 0.1)
        
        vdw = model.cal_vdw_energy(dm, sum_vdw_radius, vdw_A, vdw_N, 
                                    sample['valid1'], sample['valid2']).sum()
        hbond1 = model.cal_hbond_energy(dm, sum_vdw_radius, 
                                        hbond_coeff, A_int[:,1]).sum()
        hbond2 = model.cal_hbond_energy(dm, sum_vdw_radius, 
                                        hbond_coeff, A_int[:,-1]).sum()
        hydrophobic = model.cal_hydrophobic_energy(dm, sum_vdw_radius, 
                                            hydrophobic_coeff, A_int[:,-2]).sum()

        #constraint
        internal_vdw = cal_internal_vdw_energy(dm_internal, topological_dm,
                                                    epsilon, sigma)
        dev_fix_distance = torch.pow(initial_dm_internal-dm_internal,2).squeeze()
        dev_fix_distance = (dev_fix_distance*fix_pair).sum()
        
        if iter==0:
            initial_internal_vdw = internal_vdw.detach()
            initial_pred = torch.stack([vdw, hbond1, hbond2, hydrophobic])
            initial_pos1 = pos1.clone().detach() 

        #loss
        loss = (vdw+hbond1+hbond2+hydrophobic).sum()
        loss = loss + torch.max(internal_vdw, initial_internal_vdw)
        loss = loss + dev_fix_distance
        loss.backward()
        optimizer.step()
        #print (iter, vdw+hbond1+hbond2+hydrophobic, internal_vdw, dev_fix_distance)
    
<<<<<<< HEAD
    pos1 = pos1.data.cpu().numpy()[0]
    initial_pos1 = initial_pos1.data.cpu().numpy()[0]
    pred = torch.stack([vdw, hbond1, hbond2, hydrophobic])
    
    #rotor penalty
    rotor_penalty = 1+model.rotor_coeff*model.rotor_coeff*sample['rotor']
    pred = pred/rotor_penalty
    initial_pred = initial_pred/rotor_penalty
    
=======
    #rotor penalty
    rotor_penalty = 1+model.rotor_coeff*model.rotor_coeff*sample['rotor']

    lig_vdw = cal_vdw_energy(dm, sum_vdw_radius, vdw_A, vdw_N, is_last=True)
    lig_hbond1 = cal_hbond_energy(dm, sum_vdw_radius, hbond_coeff, A_int[:,1], is_last=True)
    lig_hbond2 = cal_hbond_energy(dm, sum_vdw_radius, hbond_coeff, A_int[:,-1], is_last=True)
    lig_hydrophobic = cal_hydrophobic_energy(dm, sum_vdw_radius, hydrophobic_coeff, A_int[:,-2], is_last=True)
    lig_energy = lig_vdw+lig_hbond1+lig_hbond2+lig_hydrophobic

    pos1 = pos1.data.cpu().numpy()[0]
    initial_pos1 = initial_pos1.data.cpu().numpy()[0]
    pred = torch.stack([vdw, hbond1, hbond2, hydrophobic])
    pred = pred/rotor_penalty
>>>>>>> 51954eaec940f7aee4841d882309a0f340a6585f
    pred = pred.data.cpu().numpy()
    initial_pred = initial_pred.data.cpu().numpy()
    extra_data = {'Initial prediction': f'{np.sum(initial_pred):.3f} Kcal/mol',
                  'Delta prediction': f'{np.sum(pred)-np.sum(initial_pred):.3f} Kcal/mol',
                  'Initial internal vdw': f'{initial_internal_vdw.item():.3f}',
                  'Final internal vdw': f'{internal_vdw.item():.3f}',
                  'Final dev fix distance': f'{dev_fix_distance.item():.3f}',
                  'ligand pos change': f'{(np.abs(pos1-initial_pos1)).sum().item():.3f}',
                  }
    end = time.time()
    
    write(of, model, pred, end-st, args, extra_data)
    write_molecule(loof, m1, pos1)

    return lig_energy

def predict(model, lf, pf, of, args, device):
    st = time.time()

    #read ligand and protein. Then, convert to rdkit object
    m1 = utils.read_molecule(lf)
    m2 = utils.extract_binding_pocket(m1, pf)

    #preprocess: convert rdkit mol obj to feature
    sample = dataset.mol_to_feature(m1, m1, m2, None, 0.0)
    sample['affinity'] = 0.0
    sample['key'] = 'None'
    sample = dataset.tensor_collate_fn([sample])
    sample = utils.dic_to_device(sample, device)

    #run prediction
    pred, _, _ = model(sample, cal_der_loss=False)
    pred = pred.data.cpu().numpy()[0]
    end = time.time()
    
    write(of, model, pred, end-st, args)
    return

if __name__=="__main__":
    #argument
    args = arguments.parser(sys.argv)

    #model
    cmd = utils.set_cuda_visible_device(args.ngpu)
    os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
    if args.potential=='morse': model = model.DTILJ(args)
    elif args.potential=='morse_all_pair': model = model.DTILJAllPair(args)
    elif args.potential=='harmonic': model = model.DTIHarmonic(args)
    elif args.potential=='harmonic_interaction_specified': model = model.DTIHarmonicIS(args)
    else: 
        print (f'No {args.potential} potential')
        exit(-1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device, args.restart_file)
    model.eval()

    if args.local_opt:            
        for lf, pf, of, loof in zip(args.ligand_files, args.protein_files, 
                args.output_files, args.ligand_opt_output_files):
            lig_energy = local_optimize(model, lf, pf, of, loof, args, device)     
            if args.ligand_prop:
                for e in lig_energy:
                    print(f"{float(e):.2f}", end=' ')
        
    else:
        for lf, pf, of in zip(args.ligand_files, args.protein_files, args.output_files):
            predict(model, lf, pf, of, args, device)     
