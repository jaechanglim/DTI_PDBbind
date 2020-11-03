import glob

"""
--data_dir=/home/wykgroup/jaechang/work/ML/PDBbind_DTI/data_pdbbind_v2019/data/ \
--filename=/home/wykgroup/jaechang/work/ML/PDBbind_DTI/data_pdbbind_v2019/pdb_to_affinity.txt \
--key_dir=../keys_pdbbind_v2019/
"""

refined = glob.glob('../pdbbind_v2019_refined/data_preprocessed/????')
# refined = sorted(refined)
with open('../index_pdbbind_v2019/index/INDEX_refined_data.2019', 'r') as r:
    r = r.readlines()[6:]
    ptoa = {rr.split()[0]:rr.split()[3] for rr in r}

ptoa = dict(sorted(ptoa.items(), key=lambda x:float(x[1])))
total = glob.glob("./data/????_out_*")
total = [t.split("/")[-1] for t in total]
total_dict = {k:ptoa[k.split("_")[0]] for k in total}
total_dict = dict(sorted(total_dict.items(), key=lambda x:float(x[1])))
with open('./pdb_to_affinity.txt', 'w') as w:
    for pa in total_dict.items():
        w.write(f"{pa[0]}\t{pa[1]}\n")


