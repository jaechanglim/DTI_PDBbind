import os
import sys
import time

p = "harmonic"
ncpu = 7
epoch = 1000
data = "/home/wykgroup/mseok/data/DTI_PDBbind"
casf_scoring = "/home/wykgroup/mseok/data/DTI_PDBbind/CASF-2016/scoring"
casf_docking = "/home/wykgroup/mseok/data/DTI_PDBbind/CASF-2016/docking"
test = "origin"

if not os.path.exists(str(test)):
    os.mkdir(str(test))
for t in ["csar1", "csar2", "scoring", "docking"]:
    if t == "docking":
        ngpu = 1
    else:
        ngpu = 0
    # if os.path.exists(f"{test}/result_{t}_{p}_{epoch}"):
    #     continue
    print(t, test, epoch)
    command = f"OMP_NUM_THREADS={ncpu} python -u ../test.py --batch_size=64 " +\
              f"--num_workers=0 --restart_file=../results/save_{epoch}.pt " +\
              f"--n_gnn=3 --dim_gnn=128 " +\
              f"--test_result_filename={test}/result_{t}_{p}_{epoch} " +\
              f"--ngpu={ngpu} --interaction_net --potential=\"{p}\" "
    if t == "csar2":
        command += f"--data_dir={data}/jaechang_csar2/data " +\
            f"--filename={data}/jaechang_csar2/pdb_to_affinity.txt " +\
            f"--key_dir={data}/jaechang_csar2/keys/"
    elif t == "csar1":
        command += f"--data_dir={data}/jaechang_csar1/data " +\
            f"--filename={data}/jaechang_csar1/pdb_to_affinity.txt " +\
            f"--key_dir={data}/jaechang_csar1/keys "
    elif t == "scoring":
        command += f"--data_dir={casf_scoring}/data " +\
            f"--filename={casf_scoring}/pdb_to_affinity.txt " +\
            f"--key_dir={casf_scoring}/keys "
    elif t == "docking":
        command += f"--data_dir={casf_docking}/data " + \
            f"--filename={casf_docking}/pdb_to_affinity.txt " +\
            f"--key_dir={casf_docking}/keys "
    command += f" > {test}/test_{t}_{p}_{epoch}"
    lines = f"""#!/bin/bash
#PBS -N mseok_{t}_{p}_{test}_{epoch}
#PBS -l nodes=1:ppn={ncpu}
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`
date

source ~/.bashrc
conda activate mseok

{command}"""
    with open(f"./jobscript_{t}.x", "w") as w:
        w.write(lines)
    print(command)
    # os.system("qsub jobscript.x")
    continue
    if t == "docking":
        time.sleep(10)
