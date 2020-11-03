import pickle
import sys
import os
import glob
import time
from multiprocessing import Pool

import numpy as np
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from scipy import sparse
from scipy.spatial import distance_matrix
from Bio.PDB import *
from Bio.PDB.PDBIO import Select


def remove_water(m):
    from rdkit.Chem.SaltRemover import SaltRemover
    remover = SaltRemover(defnData="[O]")
    return remover.StripMol(m)


def count_residue(structure):
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                count += 1
    return count


def extract(ligand, pdb):
    parser = PDBParser()
    if not os.path.exists(pdb):
        print("AAAAAAAAAAA")
        return None
    structure = parser.get_structure("protein", pdb)
    ligand_positions = ligand.GetConformer().GetPositions()

    # Get distance between ligand positions (N_ligand, 3) and
    # residue positions (N_residue, 3) for each residue
    # only select residue with minimum distance of it is smaller than 5A
    class ResidueSelect(Select):
        def accept_residue(self, residue):
            residue_positions = np.array([np.array(list(atom.get_vector())) \
                for atom in residue.get_atoms() if "H" not in atom.get_id()])
            if len(residue_positions.shape) < 2:
                print(residue)
                return 0
            min_dis = np.min(
                distance_matrix(residue_positions, ligand_positions))
            if min_dis < 5.0:
                return 1
            else:
                return 0

    io = PDBIO()
    io.set_structure(structure)
    fn = "BS_tmp_" + str(np.random.randint(0, 1000000, 1)[0]) + ".pdb"
    io.save(fn, ResidueSelect())
    m2 = Chem.MolFromPDBFile(fn)
    os.system("rm -f " + fn)
    return m2


def uff(m):
    Chem.SanitizeMol(m)
    m = Chem.AddHs(m)
    cids = AllChem.EmbedMultipleConfs(
        m,
        numConfs=20,
    )
    cenergy = []
    for conf in cids:
        converged = not AllChem.UFFOptimizeMolecule(m, confId=conf)
        cenergy.append(
            AllChem.UFFGetMoleculeForceField(m, confId=conf).CalcEnergy())
    min_idx = cenergy.index(min(cenergy))
    m = Chem.RemoveHs(m)
    sdf_name = "sdf_tmp_" + str(np.random.randint(0, 1000000, 1)[0]) + ".pdb"
    w = Chem.SDWriter(sdf_name)
    w.write(m, min_idx)
    w.close()
    retval = Chem.SDMolSupplier(sdf_name)[0]
    os.system("rm -f " + sdf_name)
    return retval


def preprocessor(l):
    pdb_fn = "_".join(l.split("/")[-1].split(".")[0].split("_")[:-2])
    key = pdb_fn.split("_")[1]
    # origin
    # l = "STOCK4S-00529_4zl4_out_1.pdb"
    data_dir = "./"
    if os.path.exists(f"{data_dir}/{key}"):
        return
    ligand_pdb_fn = l
    pdb_dir = "../../refined_set"
    bs_pdb_fn = f"{pdb_dir}/{key}/{key}_protein.pdb"

    # print("ligand_sdf_fn")
    m1 = Chem.MolFromPDBFile(ligand_pdb_fn)
    # print(m1)

    if m1 is None:
        for i in range(2, 10):
            ligand_pdb_fn[-5] = str(i)
            if os.path.exists(l):
                m1 = Chem.MolFromPDBFile(ligand_pdb_fn)
                if m1 is not None:
                    break
        print(f"{pdb_fn} no mol generated from pdb")
        return

    m1_uff = uff(m1)
    if m1_uff is None:
        print(f"{pdb_fn} no uff mol from ligand mol!")
        return

    # extract binding pocket
    m2 = extract(m1, bs_pdb_fn)
    if m2 is None:
        print(f"{pdb_fn} no extracted binding pocket!")
        return
    m2 = remove_water(m2)

    if len(m1.GetConformers()) == 0:
        return
    if len(m2.GetConformers()) == 0:
        return
    with open(data_dir + pdb_fn, "wb") as fp:
        pickle.dump((m1, m1_uff, m2, []), fp, pickle.HIGHEST_PROTOCOL)
    return


def run(l):
    try:
        return preprocessor(l)
    except:
        return


total = glob.glob("../result_pdb/*_out_1.pdb")
keys = total
for key in keys:
    run(key)
