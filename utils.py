import torch
import torch.nn as nn


# Soojung edit start
def loss_var(pred_var, pred, affinity, log=True):
    mse = (pred - affinity) ** 2
    if log:
        loss = mse * 0.5 * torch.exp(-pred_var)
        loss += 0.5 * pred_var
    else:
        loss = mse * 0.5 / pred_var
        loss += 0.5 * torch.log(pred_var)
    return loss.mean(-1)


def dic_to_device(dic, device):
    for dic_key, dic_value in dic.items():
        if isinstance(dic_value, torch.Tensor):
            dic_value = dic_value.to(device)
            dic[dic_key] = dic_value
    
    return dic

def load_data(filename):
    data = dict()
    with open(filename) as f:
        lines = f.readlines()[1:]
        lines = [l.strip() for l in lines]
        for l in lines:
            l = l.split(',')
            if len(l)==5: l.append(-100.0)
            k = l[1]
            if k not in data.keys(): data[k] = []
            data[k].append((int(l[2]),int(l[3]),l[4],float(l[5])))

    return data
        
def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    import numpy as np
    empty = []
    if ngpus>0:
        fn = f'/tmp/empty_gpu_check_{np.random.randint(0,10000000,1)[0]}'
        for i in range(4):
            os.system(f'nvidia-smi -i {i} | grep "No running" | wc -l > {fn}')
            with open(fn) as f:
                out = int(f.read())
            if int(out)==1:
                empty.append(i)
            if len(empty)==ngpus: break
        if len(empty)<ngpus:
            print ('avaliable gpus are less than required', len(empty), ngpus)
            exit(-1)
        os.system(f'rm -f {fn}')        
    
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','

    return cmd

# Soojung edit start
def initialize_model(model, device, load_save_file=False, file_path=''):
    if load_save_file:
        if device.type=='cpu':
            model.load_state_dict(torch.load(file_path, map_location='cpu'), strict=False)
        else:
            model.load_state_dict(torch.load(file_path), strict=False)
    # Soojung edit end
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    return model

def read_data(filename, key_dir):
    import pickle
    with open(filename) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        id_to_y = {l[0]:float(l[1]) for l in lines}
    with open(f'{key_dir}/train_keys.pkl', 'rb') as f:
        train_keys = pickle.load(f)
    with open(f'{key_dir}/test_keys.pkl', 'rb') as f:
        test_keys = pickle.load(f)
    return train_keys, test_keys, id_to_y        

def get_dataset_dataloader(train_keys, test_keys, data_dir, id_to_y, 
                batch_size, num_workers, pos_noise_std):
    from torch.utils.data import DataLoader
    from dataset import MolDataset, tensor_collate_fn
    train_dataset = MolDataset(train_keys, data_dir, id_to_y, 
                               pos_noise_std=pos_noise_std)
    train_dataloader = DataLoader(train_dataset,
                                   batch_size,
                                   num_workers=num_workers,
                                   collate_fn=tensor_collate_fn,
                                   shuffle=True)

    test_dataset = MolDataset(test_keys, data_dir, id_to_y)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size,
                                  num_workers=num_workers,
                                  collate_fn=tensor_collate_fn,
                                  shuffle=False)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

# Soojung edit
def write_result_var(filename, pred, true, var):
    with open(filename, 'w')  as w:
        for k in pred.keys():
            w.write(f'{k}\t{true[k]:.3f}\t')
            w.write(f'{pred[k].sum():.3f}\t')
            w.write(f'{var[k].sum():.3f}\t')
            w.write(f'{0.0}\t')
            for j in range(pred[k].shape[0]):
                w.write(f'{pred[k][j]:.3f}\t')
            w.write('\n')
    return

def write_result(filename, pred, true):
    with open(filename, 'w')  as w:
        for k in pred.keys():
            w.write(f'{k}\t{true[k]:.3f}\t')
            w.write(f'{pred[k].sum():.3f}\t')
            w.write(f'{0.0}\t')
            for j in range(pred[k].shape[0]):
                w.write(f'{pred[k][j]:.3f}\t')
            w.write('\n')
    return

def extract_binding_pocket(ligand, pdb):
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.PDBIO import Select
    import os
    import numpy as np
    from rdkit import Chem
    from scipy.spatial import distance_matrix

    parser = PDBParser()
    if not os.path.exists(pdb) : 
        #print ('AAAAAAAAAAA')
        return None
    structure = parser.get_structure('protein', pdb)
    #print (count_residue(structure))
    ligand_positions = ligand.GetConformer().GetPositions()

    class GlySelect(Select):
        def accept_residue(self, residue):
            residue_positions = np.array([np.array(list(atom.get_vector())) \
                for atom in residue.get_atoms() if 'H' not in atom.get_id()])
            #print (residue_positions)
            min_dis = np.min(distance_matrix(residue_positions, ligand_positions))
            if min_dis < 5.0:
                return 1
            else:
                return 0
    io = PDBIO()
    io.set_structure(structure)

    np.random.seed()
    fn = 'BS_tmp_'+str(np.random.randint(0,100000,1)[0])+'.pdb'
    #fd, fpath = tempfile.mkstemp(prefix='BS_tmp', dir=os.getcwd(), text=True)
    io.save(fn, GlySelect())
    #structure = parser.get_structure('protein', fn)
    #if count_residue(structure)<10: return None
    #print (count_residue(structure))
    m2 = Chem.MolFromPDBFile(fn)
    #os.unlink(fpath)
    os.system('rm -f ' + fn)

    return m2

def read_molecule(filename):
    from rdkit import Chem
    if filename[-4:]=='.sdf':
        return Chem.SDMolSupplier(filename)[0]
    elif filename[-5:]=='.mol2':
        return Chem.MolFromMol2File(filename)
    else:
        print (f'{filename} is wrong filename')
    exit(-1)
    return None
