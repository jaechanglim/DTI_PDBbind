from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import utils
import numpy as np
import torch
import random
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import Chem
from ase.io import read
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import random
import pickle
from ase import Atoms, Atom
from rdkit.Chem.rdmolops import CombineMols
from rdkit.Chem.rdmolops import SplitMolByPDBResidues
import copy
random.seed(0)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def atom_feature(m, atom_i, i_donor, i_acceptor):

    atom = m.GetAtomWithIdx(atom_i)
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28

def rotate(molecule, angle, axis, fix_com=False):
    c = molecule.GetConformers()[0]
    d = np.array(c.GetPositions())
    ori_mean = np.mean(d, 0, keepdims=True)
    if fix_com:
        d = d - ori_mean
    atoms = []
    for i in range(len(d)):
        atoms.append(Atom('C', d[i]))
    atoms = Atoms(atoms)        
    atoms.rotate(angle, axis)
    new_d = atoms.get_positions()
    if fix_com:
        new_d += ori_mean
    for i in range(molecule.GetNumAtoms()):  
        c.SetAtomPosition(i,new_d[i])
    return molecule        

def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(atom_feature(m, i, None, None))
    H = np.array(H)        
    if is_ligand:
        H = np.concatenate([H, np.zeros((n,28))], 1)
    else:
        H = np.concatenate([np.zeros((n,28)), H], 1)
    return H        

def dm_vector(d1, d2):
    n1 = len(d1)
    n2 = len(d2)
    d1 = np.repeat(np.expand_dims(d1, 1), n2, 1)
    d2 = np.repeat(np.expand_dims(d2, 0), n1, 0)
    return d1-d2

def extract_valid_amino_acid(m, amino_acids):
    ms = SplitMolByPDBResidues(m)
    valid_ms = [ms[k] for k in ms.keys()]
    #valid_ms = [ms[k] for k in ms.keys() if k in amino_acids]
    ret_m = None
    for i in range(len(valid_ms)):
        if i==0:
            ret_m = valid_ms[0]
        else:
            ret_m = CombineMols(ret_m, valid_ms[i])
    return ret_m
                
class MolDataset(Dataset):

    def __init__(self, keys, data_dir, id_to_y, random_rotation = 0.0):
        self.keys = keys
        self.data_dir = data_dir
        #self.id_to_ligand = id_to_ligand
        #self.id_to_protein = id_to_protein
        self.id_to_y = id_to_y
        self.random_rotation = random_rotation
    
        self.amino_acids = ['ALA','ARG','ASN','ASP','ASX','CYS','GLU','GLN','GLX',\
                   'GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',\
                   'THR','TRP','TYR','VAL']

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with open(self.data_dir+'/'+key, 'rb') as f:
            m1, m2 = pickle.load(f)
        
        #Remove hydrogens
        m1 = Chem.RemoveHs(m1)
        m2 = Chem.RemoveHs(m2)

        #extract valid amino acids
        m2 = extract_valid_amino_acid(m2, self.amino_acids)
        #if m2 is None : return None
        if m2 is None: print (key)    
        
        #random rotation
        angle = np.random.uniform(0,360,1)[0]
        axis = np.random.uniform(-1,1,3)
        m1 = rotate(m1, angle, axis, False)
        m2 = rotate(m2, angle, axis, False)
        
        angle = np.random.uniform(0,360,1)[0]
        axis = np.random.uniform(-1,1,3)
        m1_rot = rotate(copy.deepcopy(m1), angle, axis, True)
        
        affinity = self.id_to_y[key]
   
        #prepare ligand
        n1 = m1.GetNumAtoms()
        d1 = np.array(m1.GetConformers()[0].GetPositions())
        d1_rot = np.array(m1_rot.GetConformers()[0].GetPositions())
        adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
        h1 = get_atom_feature(m1, True)

        #prepare protein
        n2 = m2.GetNumAtoms()
        c2 = m2.GetConformers()[0]
        d2 = np.array(c2.GetPositions())
        adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
        h2 = get_atom_feature(m2, False)
            
        #prepare distance vector
        dmv = dm_vector(d1,d2)
        dmv_rot = dm_vector(d1_rot,d2)

        #node indice for aggregation
        valid = np.ones((n1+n2,))
        #pIC50 to class
        #Y = 1 if Y > 6 else 0
        affinity = -affinity
        
        #if n1+n2 > 300 : return None
        sample = {
                  'h1':h1, \
                  'adj1': adj1, \
                  'h2':h2, \
                  'adj2': adj2, \
                  'dmv': dmv, \
                  'dmv_rot': dmv_rot, \
                  'valid': valid, \
                  'affinity': affinity, \
                  'key': key, \
                  }

        return sample

class DTISampler(Sampler):

    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self):
        #return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples

def my_collate_fn(batch):
    n_valid_items = len([0 for item in batch if item is not None])
    max_natoms1 = max([len(item['h1']) for item in batch if item is not None])
    max_natoms2 = max([len(item['h2']) for item in batch if item is not None])
    
    h1 = np.zeros((n_valid_items, max_natoms1, 56))
    h2 = np.zeros((n_valid_items, max_natoms2, 56))
    adj1 = np.zeros((n_valid_items, max_natoms1, max_natoms1))
    adj2 = np.zeros((n_valid_items, max_natoms2, max_natoms2))
    dmv = np.zeros((n_valid_items, max_natoms1, max_natoms2, 3))
    dmv_rot = np.zeros((n_valid_items, max_natoms1, max_natoms2, 3))
    valid = np.zeros((n_valid_items, max_natoms1+max_natoms2))
    affinity = np.zeros((n_valid_items,))
    keys = []
    i = 0
    for j in range(len(batch)):
        if batch[j] is None : continue
        natom1 = len(batch[j]['h1'])
        natom2 = len(batch[j]['h2'])
        
        h1[i,:natom1] = batch[j]['h1']
        adj1[i,:natom1,:natom1] = batch[j]['adj1']
        h2[i,:natom2] = batch[j]['h2']
        adj2[i,:natom2,:natom2] = batch[j]['adj2']
        dmv[i,:natom1,:natom2] = batch[j]['dmv']
        dmv_rot[i,:natom1,:natom2] = batch[j]['dmv_rot']
        valid[i,:natom1+natom2] = batch[j]['valid']
        affinity[i] = batch[j]['affinity']
        keys.append(batch[j]['key'])
        i+=1

    h1 = torch.from_numpy(h1).float()
    adj1 = torch.from_numpy(adj1).float()
    h2 = torch.from_numpy(h2).float()
    adj2 = torch.from_numpy(adj2).float()
    dmv = torch.from_numpy(dmv).float()
    dmv_rot = torch.from_numpy(dmv_rot).float()
    valid = torch.from_numpy(valid).float()
    affinity = torch.from_numpy(affinity).float()
    
    return h1, adj1, h2, adj2, dmv, dmv_rot, valid, affinity, keys
