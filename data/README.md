# DTI\_PDBbind\_preprocess
Data preprocessing codes for DTI\_PDBbind.

Follow `README.md` to preprocess the data.


`pdbbind_v2019_random_screening/chembl_27_chemreps.txt` and `pdbbind_v2019_random_screening/id_smiles.txt` are downloaded from [CHEMBL](https://chembl.gitbook.io/chembl-interface-documentation/) and [IBS](https://www.ibscreen.com), respectively.

## Workflow
Do preprocess as following order:

0. [Download](#download): Download all necessary data.
1. [rcsb_pdb](#rcsb_pdb): Download ligand sdfs from [rcsb](https://www.rcsb.org) via crawling.
2. [pdbbind_v2019_refined](#pdbbind_v2019_refined): Main dataset.
3. [pdbbind_v2019_docking_nowater](#pdbbind_v2019_docking_nowater): Docking dataset.
4. [pdbbind_v2019_random_screening](#pdbbind_v2019_random_screening): Random screening dataset.
5. [pdbbind_v2019_cross_screening](#pdbbind_v2019_cross_screening): Cross screening dataset.
6. [CASF-2016](#CASF-2016): CASF-2016 benchmark dataset for testing the model.
7. [csar1](#csar1): CSAR_NRC_HiQ_Set1 for testing the model.
8. [csar2](#csar2): CSAR_NRC_HiQ_Set2 for testing the model.

### Download
Before start, you should download following datasets.
- pdbbind\_v.2019\_refined\_set from [pdbbind](http://www.pdbbind-cn.org)
- CSAR\_NRC\_HiQ\_Set from [csardock](http://www.csardock.org)
- CASF-2016 from [pdbbind](http://www.pdbbind-cn.org/casf.php)

After downloading each dataset, use `tar -xvzf` to unzip the `tar.gz` files.

### rcsb_pdb
- We had already downloaded and preprocessed ligand sdf from [rcsb](https://www.rcsb.org). The sdfs are prepared in `rcsb_pdb/refined_data` directory, so you can skip the `rcsb_pdb` process.

### pdbbind_v2019_refined
Change directory to `pdbbind_v2019_refined`.
> Train data

To prepare train data, change directory to `pp_train`, and execute `preprocess.py`.
> Test data

To prepare test data, change directory to `pp_test`, and execute `preprocess.py`.
*Important:* After preprocessing all data, you should gather them into `data` directory.
> Keys

To generate keys for the preprocessed data in `data` directory, change directory to `keys`, and execute `generate_keys.py`.
> pdb\_to\_affinity

To generate pdb\_to\_affinity.txt which contains binding affinity value for each key, execute `pdb_to_affinity.py`.

### pdbbind_v2019_docking_nowater
Change directory to `pdbbind_v2019_docking_nowater`.
> Decoy ligands
1. To prepare the decoy ligands, execute `cal_docking.py`. This perform docking simulation and generates `pdbqt` files in `result_pdbqt` directory.
2. After docking calculation, execute `pdbqt_to_pdb.py`. This splits the `pdbqt` file into several `pdb` files which will be saved in `result_pdb` directory.
3. To preprocess files which are generated from docking simulation, change directory to `data`, and execute `preprocess.py`.
> Keys

To generate keys for the preprocessed data in `data` directory, change directory to `keys`, and execute `generate_keys.py`.
> pdb\_to\_affinity

To generate pdb\_to\_affinity.txt which contains binding affinity value for each key, execute `pdb_to_affinity.py`.

### pdbbind_v2019_random_screening
Change directory to `pdbbind_v2019_random_screening` and execute in the order you executed in the `pdbbind_v2019_docking_nowater`.

### pdbbind_v2019_cross_screening
Change directory to `pdbbind_v2019_cross_screening` and execute in the order you executed in the `pdbbind_v2019_docking_nowater`.

### CASF-2016
Change directory to `CASF-2016`
> Split the docking decoys from the mol2 files.

To split the docking decoys from the mol2 files, change directory to `mol2_decoy_docking` and execute `run_split.py`
> Split the screening decoys from the mol2 files.

To split the screening decoys from the mol2 files, change directory to `mol2_screening_docking` and execute `run_split.py`
> scoring test data preprocessing

To preprocess the scoring test data, change directory to `scoring/data`, and execute `preprocess.py`.
1. Keys

To generate keys for the preprocessed data in `data` directory, change directory to `scoring/keys`, and execute `generate_keys.py`.

2. pdb\_to\_affinity

To generate pdb\_to\_affinity.txt which contains binding affinity value for each key, change directory to `scoring`, and execute `pdb_to_affinity.py`.
> docking test data preprocessing

To preprocess the docking test data, change directory to `docking/data`, and execute `preprocess.py`.
1. Keys

To generate keys for the preprocessed data in `data` directory, change directory to `docking/keys`, and execute `generate_keys.py`.

2. pdb\_to\_affinity

To generate pdb\_to\_affinity.txt which contains binding affinity value for each key, change directory to `docking`, and execute `pdb_to_affinity.py`.
> screening test data preprocessing

To preprocess the screening test data, change directory to `screening/data`, and execute `preprocess.py`.
1. Keys

To generate keys for the preprocessed data in `data` directory, change directory to `screening/keys`, and execute `generate_keys.py`.

2. pdb\_to\_affinity

To generate pdb\_to\_affinity.txt which contains binding affinity value for each key, change directory to `screening`, and execute `pdb_to_affinity.py`.

### csar1
Change directory to `csar1`.
> Split ligand and protein from each complex

To splilt ligand and protein from each complex of CSAR_NRC_HiQ_Set1 dataset, execute `split_receptor_ligand.py`.
> Keys

To generate keys for the preprocessed data in `data` directory, change directory to `keys`, and execute `generate_keys.py`.
> pdb\_to\_affinity

To generate pdb\_to\_affinity.txt which contains binding affinity value for each key, execute `pdb_to_affinity.py`.

### csar2
Change directory to `csar2` and execute in the order you executed in the `csar1`.
