import glob
import sys

#read rmsd
rmsd_filenames = glob.glob('/home/wykgroup/jaechang/work/ML/PDBbind_DTI/data_casf2013_decoy_docking/*_rmsd.dat')
id_to_rmsd1, id_to_rmsd2 = dict(), dict()
for fn in rmsd_filenames:
    with open(fn) as f:
        lines = f.readlines()
    lines = [l.strip().split() for l in lines]
    for l in lines:
        id_to_rmsd1[l[0]] = float(l[1])
        id_to_rmsd2[l[0]] = float(l[2])

#read data
decoy_filenames = glob.glob(sys.argv[1])
decoy_filenames = sorted(decoy_filenames, key=lambda x:int(x.split('_')[-1]))
ref_filename = sys.argv[2]
with open(ref_filename) as f:
    lines = f.readlines()
    lines = [l.split() for l in lines]
    ref_to_pred = {l[0]:float(l[2]) for l in lines}
    for l in lines: 
        id_to_rmsd1[l[0]]=0.0
        id_to_rmsd2[l[0]]=0.0

for fn in decoy_filenames:
    with open(fn) as f:
        lines = f.readlines()
        lines = [l.strip().split() for l in lines]
    #id to affinity
    id_to_pred = {l[0]:float(l[2]) for l in lines}

    #existing pdb
    pdbs = sorted(list(set([k.split()[0].split('_')[0] for k in id_to_pred.keys()])))
    pdb_success=[]
    for pdb in pdbs:
        if pdb not in ref_to_pred.keys(): continue
        selected_keys = [k for k in id_to_pred.keys() if pdb in k]
        pred = [id_to_pred[k] for k in selected_keys]

        selected_keys.append(pdb)
        pred.append(ref_to_pred[pdb])

        pred, sorted_keys = zip(*sorted(zip(pred, selected_keys)))
        rmsd = [id_to_rmsd2[k] for k in sorted_keys]
        top_n_success = []
        #print (pdb, sorted_keys[:3], pred[:3], rmsd[:3])
        for top_n in [1,2,3]:
            if min(rmsd[:top_n])<2.0:
                top_n_success.append(1)
            else:             
                top_n_success.append(0)
        pdb_success.append(top_n_success)                
    
    print (fn, end='\t')
    for top_n in [1,2,3]:
        success = [s[top_n-1] for s in pdb_success]
        print (f'{sum(success)/len(success):.3f}', end='\t')
    print ()
