import sys
import glob

#make cluster
with open('/home/jaechang/work/jaechang_horus/work/ML/PDBbind_DTI/DTI_PDBbind/casf2013_benchmark/index.txt') as f:
    lines = f.readlines()
pdbs = [l.split()[0].lower() for l in lines]    
clusters = [pdbs[i*3:i*3+3]for i in range(65)]

#filename
filename = 'test.txt'
if len(sys.argv)>1: filename = sys.argv[1]
filenames = glob.glob(filename)
filenames = sorted(filenames, key=lambda x:int(x.split('_')[-1]))
for fn in filenames:
    with open(fn) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        pdb_to_pred = dict({l[0]:-float(l[2]) for l in lines})

    high_success = []
    low_success = []
    for cluster in clusters:
        no_data = False
        for c in cluster: 
            if c not in pdb_to_pred.keys(): no_data=True
        if no_data: continue        

        preds = [pdb_to_pred[p] for p in cluster]
        preds, ordered_pdb = zip(*sorted(zip(preds, cluster[:])))
        
        if ordered_pdb[2] == cluster[2]: 
            low_success.append(1)
        else:        
            low_success.append(0)
        
        if ordered_pdb[0] == cluster[0] and ordered_pdb[1] == cluster[1] \
                and ordered_pdb[2] == cluster[2]: 
            high_success.append(1)
        else:        
            high_success.append(0)

    print (fn, end = '\t')
    print (f'{sum(high_success)/len(high_success):.3f}', len(high_success), end='\t')
    print (f'{sum(low_success)/len(low_success):.3f}', len(low_success), end='\t')
    print () 



