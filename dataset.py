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
from rdkit.Chem import rdFreeSASA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem import AllChem
import copy
random.seed(0)

interaction_types = ['saltbridge', 'hbonds', 'pication', 
        'pistack', 'halogen', 'waterbridge', 'hydrophobic', 'metal_complexes']

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:"\
                        .format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(m, atom_i, i_donor, i_acceptor):
    atom = m.GetAtomWithIdx(atom_i)
    symbol_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']
    symbol = atom.GetSymbol()
    degree = atom.GetDegree()
    total_numH = atom.GetTotalNumHs()
    valance = atom.GetImplicitValence()
    return np.array(one_of_k_encoding_unk(symbol, symbol_list) +
                    one_of_k_encoding_unk(degree, [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(total_numH, [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(valence, [0, 1, 2, 3, 4, 5]) +
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
    # of atoms in molecule1, molecule2 = n1, n2
        shape of d1, d2: [n1, 3], [n2, 3]
    repeat these vectors, make vector shape of [n1, n2, 3] subtract two vectors
    Square value of resulting vector means distances between every atoms in two
    molecules
    :param d1: position vector of atoms in molecule1, shape: [n1, 3]
    :param d2: position vector of atoms in molecule2, shape: [n2, 3]
    :return: subtraction between enlarged vectors from d1, d2
        shape: [n1, n2, 3]
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
    :return: single molecule's information which
             all of amino acid residues are combined
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

def position_to_index(positions, target_position):
    indice = np.where(np.all(positions==target_position,axis=1))[0]
    diff = positions-np.expand_dims(np.array(target_position), 0)
    diff = np.sum(np.power(diff, 2), -1)
    indice = np.where(diff<1e-6)[0]
    return indice.tolist()

def get_interaction_matrix(d1, d2, interaction_data):
    """
    d1: distance matrix1 that contains ligand molecule atoms' positions
        shape: [, # ligand atoms, 3]
    d2: distance matrix2 that contains protein molecule atoms' positions
        shape: [, # protein atoms, 3]
    interaction_data: list of tuples of position(x,y,z) that compose the
                      certain interaction
    interaction_types: total 7 of interaction types(global variable)
    """
    n1, n2 = len(d1), len(d2) 
    A = np.zeros((len(interaction_types), n1, n2))
    for i_type,k in enumerate(interaction_types):
        for ps in interaction_data[k]:
            p1, p2 = ps
            i1 = position_to_index(d1, p1)
            i2 = position_to_index(d2, p2)
            if len(i1)==0:
                i1 = position_to_index(d1, p2)
                i2 = position_to_index(d2, p1)
            if len(i1)==0 or len(i2)==0:
                pass
            else:
                i1, i2 = i1[0], i2[0]
                A[i_type, i1, i2] = 1
    return A

def classifyAtoms(mol, polar_atoms=[7,8,15,16]):
    """
    Taken from
    https://github.com/mittinatten/freesasa/blob/master/src/classifier.c
    """
    symbol_radius = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, 
        "S": 1.80, "SE": 1.90,
	"F": 1.47, "CL": 1.75, "BR": 1.83, "I": 1.98,
	"LI": 1.81, "BE": 1.53, "B": 1.92,
	"NA": 2.27, "MG": 1.74, "AL": 1.84, "SI": 2.10,
	"K": 2.75, "CA": 2.31, "GA": 1.87, "GE": 2.11, "AS": 1.85,
	"RB": 3.03, "SR": 2.49, "IN": 1.93, "SN": 2.17, "SB": 2.06, "TE": 2.06}

    radii = [] 
    for atom in mol.GetAtoms():
        # mark everything as apolar to start
        atom.SetProp("SASAClassName", "Apolar")
        #identify polar atoms and change their marking
        if atom.GetAtomicNum() in polar_atoms:
            atom.SetProp("SASAClassName", "Polar") # mark as polar
        elif atom.GetAtomicNum() == 1:
            if atom.GetBonds()[0].GetOtherAtom(atom).GetAtomicNum() \
              in polar_atoms:
                atom.SetProp("SASAClassName", "Polar") # mark as polar
            radii.append(symbol_radius[atom.GetSymbol().upper()])
    return(radii)

def cal_sasa(m):
    radii = rdFreeSASA.classifyAtoms(m)
    radii = classifyAtoms(m)
    #radii = rdFreeSASA.classifyAtoms(m1)
    sasa=rdFreeSASA.CalcSASA(m, radii, 
            query=rdFreeSASA.MakeFreeSasaAPolarAtomQuery())
    return sasa

def get_vdw_radius(a):
    atomic_number = a.GetAtomicNum()
    atomic_number_to_radius = {6: 1.90, 7: 1.8, 8: 1.7, 16: 2.0, 15:2.1, \
            9:1.5, 17:1.8, 35: 2.0, 53:2.2}
    if atomic_number in atomic_number_to_radius.keys():
        return atomic_number_to_radius[atomic_number]
    return Chem.GetPeriodicTable().GetRvdw(atomic_number)



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
              shape: [# of ligand molecule's atom,
                      property]
             -adj1 :  adjacency matrix of ligand molecule(m1)
              shape: [# of ligand molecule's atom,
                      # of ligand molecule's atom]
             -h2 :  atom feature one-hot vector of protein moleucule(m2)
              shape: [# of protein molecule's atom,
                     property]
             -adj2 :  adjacency matrix of protein molecule(m2)
              shape: [# of protein molecule's atom,
                      # of protein molecule's atom]
             -dmv :  distance matrix vector between every atoms of m1 and m2
              shape: [# of ligand molecule's atom,
                     # of protein molecule's atom, 3]
             -dmv_rot : distance matrix vector between every atoms of rotated
                        version m1 and m2
              shape: [# of ligand molecule's atom,
                      # of protein molecule's atom,
                      3]
             -valid :   true valid atom indices which will be used after
                        'my_collate_fn' makes each molecules' vector and
                        adjacency matrices into same size with zero padding
                        and calculate property
              shape: [# of ligand molecule's atom]
             -affinity : the property value whether 1 or 0
                         -> interaction on = 1, interaction off = 0
             -key : key value of the protein-ligand interaction
    """

    def __init__(self, keys, data_dir, id_to_y, random_rotation = 0.0):
        self.keys = keys
        self.data_dir = data_dir
        self.id_to_y = id_to_y
        self.random_rotation = random_rotation
        self.amino_acids = ['ALA','ARG','ASN','ASP','ASX','CYS','GLU','GLN',
                   'GLX', 'GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO',
                   'SER', 'THR','TRP','TYR','VAL']
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        1. get each keys from train_keys, test_keys
        2. get two molecules from pickle file containing key in the name at
           data directory
        3. extract valid amino acid residue from the molecule(m2)
        4. get adjacency matrix, conformer, atom feature from two rotated
           molecules, m1 and m2
        """
        key = self.keys[idx]
        #key = '1x8r'
        with open(self.data_dir+'/'+key, 'rb') as f:
            m1, m1_uff, m2, interaction_data = pickle.load(f)
        
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
        #m1 = rotate(m1, angle, axis, False)
        #m2 = rotate(m2, angle, axis, False)
        
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

        #affinity
        affinity = -affinity

        #get interaction matrix
        A_int = get_interaction_matrix(d1, d2, interaction_data)

        #cal sasa
        sasa = cal_sasa(m1)
        dsasa = sasa-cal_sasa(m1_uff)
        
        #count rotatable bonds
        rotor = CalcNumRotatableBonds(m1)

        #charge
        AllChem.ComputeGasteigerCharges(m1)
        AllChem.ComputeGasteigerCharges(m2)
        charge1 = [float(m1.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) \
                         for i in range(m1.GetNumAtoms())]
        charge2 = [float(m2.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) \
                         for i in range(m2.GetNumAtoms())]

        #partial charge calculated by gasteiger
        charge1 = np.array(charge1)
        charge2 = np.array(charge2)

        #There is nan for some cases.
        charge1 = np.nan_to_num(charge1, nan=0, neginf=0, posinf=0)
        charge2 = np.nan_to_num(charge2, nan=0, neginf=0, posinf=0)

        #valid
        valid1 = np.ones((n1,))
        valid2 = np.ones((n2,))

        #no metal
        metal_symbols = ['Zn', 'Mn', 'Co', 'Mg', 'Ni', 'Fe', 'Ca', 'Cu']
        no_metal1 = np.array([1 if a.GetSymbol() not in metal_symbols else 0 
                for a in m1.GetAtoms()])
        no_metal2 = np.array([1 if a.GetSymbol() not in metal_symbols else 0 
                for a in m2.GetAtoms()])
        #vdw radius
        vdw_radius1 = np.array([get_vdw_radius(a) for a in m1.GetAtoms()])
        vdw_radius2 = np.array([get_vdw_radius(a) for a in m2.GetAtoms()])

        sample = {
                  'h1':h1, 
                  'adj1': adj1, 
                  'h2':h2, 
                  'adj2': adj2, 
                  'A_int': A_int, 
                  'dmv': dmv, 
                  'dmv_rot': dmv_rot, 
                  'affinity': affinity, 
                  'sasa': sasa, 
                  'dsasa': dsasa, 
                  'rotor': rotor, 
                  'charge1': charge1, 
                  'charge2': charge2, 
                  'vdw_radius1': vdw_radius1, 
                  'vdw_radius2': vdw_radius2, 
                  'valid1': valid1, 
                  'valid2': valid2, 
                  'no_metal1': no_metal1, 
                  'no_metal2': no_metal2, 
                  'key': key, 
                  }

        return sample


class DTISampler(Sampler):
    """
    Torch Sampler object that used in Data Loader.
    This simply changes the __iter__ part of the dataset class.
    In this case, we have weight parameter for each data which is importance,
    and sampling will be done by choosing each elements proportionally to this
    weight value.
    Total data size is len(weights) and sampler choose only num_samples number
    of data among total data.
    """
    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self):
        retval = np.random.choice(len(self.weights),
                                  self.num_samples,
                                  replace=self.replacement,
                                  p=self.weights) 
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples


def my_collate_fn(batch):
    """
    The function used as argument in torch dataloader.
    Since there can be many molecules with different atom number,
    to do calculation with them, we should
    make their size equally so that in this function we add zero padding to
    the small size vectors.
    The 'V' in batch have information of original size of the vector before
    zero padding.
    :param batch: containing dictionary which returned by MolDataset
    shape: [batch_size, 1] 
    :return: list of each dictionary items resized with maximum values of
    atom numbers of ligand and protein
    """ 
    n_valid_items = len([0 for item in batch if item is not None])
    max_natoms1 = max([len(item['h1']) for item in batch if item is not None])
    max_natoms2 = max([len(item['h2']) for item in batch if item is not None])
    
    h1 = np.zeros((n_valid_items, max_natoms1, 56))
    h2 = np.zeros((n_valid_items, max_natoms2, 56))
    adj1 = np.zeros((n_valid_items, max_natoms1, max_natoms1))
    adj2 = np.zeros((n_valid_items, max_natoms2, max_natoms2))
    A_int = np.zeros((n_valid_items, len(interaction_types),
        max_natoms1, max_natoms2))
    dmv = np.zeros((n_valid_items, max_natoms1, max_natoms2, 3))
    dmv_rot = np.zeros((n_valid_items, max_natoms1, max_natoms2, 3))
    affinity = np.zeros((n_valid_items,))
    sasa = np.zeros((n_valid_items,))
    dsasa = np.zeros((n_valid_items,))
    rotor = np.zeros((n_valid_items,))
    charge1 = np.zeros((n_valid_items, max_natoms1))
    charge2 = np.zeros((n_valid_items, max_natoms2))
    vdw_radius1 = np.zeros((n_valid_items, max_natoms1))
    vdw_radius2 = np.zeros((n_valid_items, max_natoms2))
    valid1 = np.zeros((n_valid_items, max_natoms1))
    valid2 = np.zeros((n_valid_items, max_natoms2))
    no_metal1 = np.zeros((n_valid_items, max_natoms1))
    no_metal2 = np.zeros((n_valid_items, max_natoms2))
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
        A_int[i,:,:natom1,:natom2] = batch[j]['A_int']
        dmv[i,:natom1,:natom2] = batch[j]['dmv']
        dmv_rot[i,:natom1,:natom2] = batch[j]['dmv_rot']
        affinity[i] = batch[j]['affinity']
        sasa[i] = batch[j]['sasa']
        dsasa[i] = batch[j]['dsasa']
        rotor[i] = batch[j]['rotor']
        charge1[i,:natom1] = batch[j]['charge1']
        charge2[i,:natom2] = batch[j]['charge2']
        vdw_radius1[i,:natom1] = batch[j]['vdw_radius1']
        vdw_radius2[i,:natom2] = batch[j]['vdw_radius2']
        valid1[i,:natom1] = batch[j]['valid1']
        valid2[i,:natom2] = batch[j]['valid2']
        no_metal1[i,:natom1] = batch[j]['no_metal1']
        no_metal2[i,:natom2] = batch[j]['no_metal2']
        keys.append(batch[j]['key'])
        i+=1

    h1 = torch.from_numpy(h1).float()
    adj1 = torch.from_numpy(adj1).float()
    h2 = torch.from_numpy(h2).float()
    adj2 = torch.from_numpy(adj2).float()
    dmv = torch.from_numpy(dmv).float()
    dmv_rot = torch.from_numpy(dmv_rot).float()
    A_int = torch.from_numpy(A_int).float()
    affinity = torch.from_numpy(affinity).float()
    sasa = torch.from_numpy(sasa).float()
    dsasa = torch.from_numpy(dsasa).float()
    rotor = torch.from_numpy(rotor).float()
    charge1 = torch.from_numpy(charge1).float()
    charge2 = torch.from_numpy(charge2).float()
    vdw_radius1 = torch.from_numpy(vdw_radius1).float()
    vdw_radius2 = torch.from_numpy(vdw_radius2).float()
    valid1 = torch.from_numpy(valid1).float()
    valid2 = torch.from_numpy(valid2).float()
    no_metal1 = torch.from_numpy(no_metal1).float()
    no_metal2 = torch.from_numpy(no_metal2).float()

    return h1, adj1, h2, adj2, A_int, dmv, dmv_rot, \
           affinity, sasa, dsasa, rotor, charge1, charge2, \
           vdw_radius1, vdw_radius2, valid1, valid2, \
           no_metal1, no_metal2, keys\

