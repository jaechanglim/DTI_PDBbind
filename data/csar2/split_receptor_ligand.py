import pickle
import glob
import os

from rdkit import Chem
from rdkit.Chem import AllChem


def split_protein_ligand(complex_pdb,
                         protein_pdb,
                         ligand_pdb,
                         ligand_sdf=None,
                         ligand_smiles=None,
                         ligand_resname=None):
    import pymol
    from pymol import cmd as pymol_cmd

    # load complex in pymol session
    pymol_cmd.load(complex_pdb, 'complex')
    pymol_cmd.remove('hydrogens')

    # extract ligand
    if ligand_resname is not None:
        pymol_cmd.extract('ligand', f'resn {ligand_resname}')
    else:
        pymol_cmd.extract('ligand', 'not polymer')

    # extract protein
    pymol_cmd.extract('receptor', 'polymer')

    # save protein
    pymol_cmd.save(protein_pdb, 'receptor')

    # save ligand
    pymol_cmd.save(ligand_pdb, 'ligand')

    # delete session
    pymol_cmd.delete('all')

    # pdb to sdf
    if ligand_sdf is not None:
        if ligand_smiles:
            try:
                m_smiles = Chem.RemoveHs(Chem.MolFromSmiles(ligand_smiles))
                m_pdb = Chem.RemoveHs(Chem.MolFromPDBFile(ligand_pdb))
                m = AllChem.AssignBondOrdersFromTemplate(m_smiles, m_pdb)
                w = Chem.SDWriter(ligand_sdf)
                w.write(m)
                w.close()
            except Exception as e:
                print(f"{ligand_sdf} failed: {e}")
                os.system(f'obabel {ligand_pdb} -O {ligand_sdf}')
        else:
            os.system(f'obabel {ligand_pdb} -O {ligand_sdf}')
    return


"""
#read ligand resname
with open('CSAR_NRC_HiQ_Set/SUMMARY_FILES/set1.csv') as f: lines1 = f.readlines()[1:]
with open('CSAR_NRC_HiQ_Set/SUMMARY_FILES/set2.csv') as f: lines2 = f.readlines()[1:]
set_and_id_to_ligand_resname = dict()
for l in lines1:
    l = l.strip().replace(' ', '').split(',')
    set_and_id_to_ligand_resname[(1, l[0])] = l[-1]
for l in lines2:
    l = l.strip().replace(' ', '').split(',')
    set_and_id_to_ligand_resname[(2, l[0])] = l[-1]
"""

# get filesnames of the complex
filenames = glob.glob(
    '../CSAR_NRC_HiQ_Set/Structures/set2/*/set2_*_complex.mol2')

with open("./csar2.txt", "r") as f:
    lines = f.readlines()
    lines = [l.split() for l in lines]
    indices = [l[0] for l in lines]
    pdbs = [l[1] for l in lines]
    ligands = [l[3] for l in lines]
    idx_to_pdb = dict(zip(indices, pdbs))
    idx_to_ligand = dict(zip(indices, ligands))

with open("../CSAR_NRC_HiQ_Set/binding.pkl", "rb") as f:
    smiles_dic = pickle.load(f)

# make directories
if not os.path.exists("./protein"):
    os.mkdir("./protein")
if not os.path.exists("./ligand"):
    os.mkdir("./ligand")

# failed = ["130", "83", "6", "116", "107"]

# run calculation
for fn in filenames[:]:
    complex_id = fn.split('/')[4]
    if complex_id not in indices:
        continue
    # if complex_id not in failed:
    #     continue

    # get id
    print(fn)
    pdb_id = idx_to_pdb[complex_id].lower()
    orig_resname = idx_to_ligand[complex_id]
    ligand_resname = "INH"
    # print(pdb_id)
    # print(ligand_resname)

    # filenames
    receptor_pdb = f'protein/{pdb_id}.pdb'
    ligand_pdb = f'ligand/{pdb_id}_{ligand_resname}.pdb'
    ligand_sdf = f'ligand/{pdb_id}_{ligand_resname}.sdf'

    # split
    # split_protein_ligand(fn, receptor_pdb, ligand_pdb,
    #                      ligand_sdf, ligand_resname=ligand_resname)
    if pdb_id.upper() == "1GX0":
        smiles = "O[C@H]1[C@@H](O)[C@@H](O[C@@H]1CO[P](O)(=O)O[P](O)(O)=O)N2C=CC(=O)NC2=O"
    else:
        smiles_list = smiles_dic[pdb_id.upper()]
        smiles_list = [smiles for smiles in smiles_list if orig_resname in smiles]
        smiles = smiles_list[0].split()[1]
    split_protein_ligand(fn,
                         receptor_pdb,
                         ligand_pdb,
                         ligand_sdf,
                         ligand_smiles=smiles,
                         ligand_resname=ligand_resname)

    # check ligand sdf file is valid
    m_sdf = Chem.SDMolSupplier(ligand_sdf)[0]
    if m_sdf is None:
        print('\n############################################################')
        print(f'{fn} is wrong')
        print('############################################################\n')
