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


def rotate(molecule, angle, axis, fix_com=False):
    """
    Since edge of each molecules are changing by different orientation,
    this funciton used to make molecules rotation-invariant and enables further
    self-supervised leanring.
    
    :param molecule: rdkit molecule object
    :param anble: angle to rotate,
                  random value between 0, 360
    :param axis: axis for rotation,
                 vector with three random values between 0, 1
                 (a, b, c) -> each values means x, y, z value of the vector
    :return: rotated molecule
    """
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
     

def dm_vector(d1, d2):
    """
    Get distances for every atoms in molecule1(ligand), molecule2(protein).
    # of atoms in molecule1, molecule2 = n1, n2 / shape of d1, d2 = [n1, 3], [n2, 3]
    repeat these vectors, make vector shape of [n1, n2, 3] subtract two vectors
    Square value of resulting vector means distances between every atoms in two molecules
    :param d1: position vector of atoms in molecule1, shape: [n1, 3]
    :param d2: position vector of atoms in molecule2, shape: [n2, 3]
    :return: subtraction between enlarged vectors from d1, d2, shape : [n1, n2, 3]
    """
    n1 = len(d1)
    n2 = len(d2)
    d1 = np.repeat(np.expand_dims(d1, 1), n2, 1)
    d2 = np.repeat(np.expand_dims(d2, 0), n1, 0)
    return d1-d2


def extract_valid_amino_acid(m, amino_acids):
    """
    Divide molecule into PDB residues and only select the residues
    belong to amino acids. Then, combine the all of the residues
    to make new molecule object. This is not 'real' molecule, just
    the information of the molecule with several residues
    :param m: rdkit molecule object
    :param amino_acids: lists of amino acids, total 22 exist
    :return: one molecule information that all of amino acid residues are combined
    """
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
    """
    Basic molecule dataset for DTI
    :param keys: each ligand-protein pair information keys
    :param data_dir: directory for ligand-protein molecule
                     that can find information with param 'keys'
    :param id_to_y: value 10 if high interaction exist else 0
                    this is dictionary with param 'keys'
    :return: dictionary
             -h1 :  atom feature one-hot vector of ligand molecule(m1)
                    shape: [# of ligand molecule's atom, property]
             -adj1 :  adjacency matrix of ligand molecule(m1)
                      shape: [# of ligand molecule's atom, # of ligand molecule's atom]
             -h2 :  atom feature one-hot vector of protein moleucule(m2)
                    shape: [# of protein molecule's atom, property]
             -adj2 :  adjacency matrix of protein molecule(m2)
                      shape: [# of protein molecule's atom, # of protein molecule's atom]
             -dmv :  distance matrix vector between every atoms of m1 and m2
                     shape: [# of ligand molecule's atom, # of protein molecule's atom, 3]
             -dmv_rot : distance matrix vector between every atoms of rotated version m1 and m2
                        shape: [# of ligand molecule's atom, # of protein molecule's atom, 3]
             -valid :   true valid atom indices which will be used after 'my_collate_fn' makes each molecules'
                        vector and adjacency matrices into same size with zero padding and calculate property
                        shape: [# of ligand molecule's atom]
             -affinity :   the property value whether 1 or 0 -> interaction on = 1, interaction off = 0
             -key : key value of the protein-ligand interaction
    """

    def __init__(self, keys, data_dir, id_to_y, random_rotation = 0.0):
        self.keys = keys
        self.data_dir = data_dir
        self.id_to_y = id_to_y
        self.random_rotation = random_rotation
        self.amino_acids = ['ALA','ARG','ASN','ASP','ASX','CYS','GLU','GLN','GLX',\
                   'GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',\
                   'THR','TRP','TYR','VAL']

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        1. get each keys from train_keys, test_keys
        2. get two molecules from pickle file containing key in the name at data directory
        3. extract valid amino acid residue from the molecule(m2)
        4. get adjacency matrix, conformer, atom feature from two rotated molecules, m1 and m2
        """
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
        
        # label
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
    """
    Torch Sampler object that used in Data Loader.
    This simply changes the __iter__ part of the dataset class.
    In this case, we have weight parameter for each data which means importance,
    and sampling will be done by choosing each elements proportionally to this
    weight value.
    Total data size is len(weights) and sampler choose only num_samples number of
    data among total data.
    """
    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self):
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples


def my_collate_fn(batch):
    """
    The function used as argument in torch dataloader.
    Since there can be many molecules with different atom number, to do calculation with them, we should
    make their size equally so that in this function we add zero padding to the small size vectors.
    The 'V' in batch have information of original size of the vector before zero padding.
    :param batch: shape: [batch_size, 1] containing dictionary which returned by MolDataset
    :return: list of each dictionary items resized with maximum values of atom numbers of ligand and protein
    """ 
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
