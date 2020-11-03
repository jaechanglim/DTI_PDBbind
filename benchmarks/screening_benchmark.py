import os
import time
import sys
import glob

epoch = 1000
ncpu = 7
dirs = list(range(6, 10))
dirs = [f"Xhb_check{dir}" for dir in dirs]
data_path ="/home/wykgroup/mseok/data/DTI_PDBbind/CASF-2016/screening"

if not os.path.exists("./screening"):
    os.mkdir("./screening")
for exp_name in dirs:
    if not os.path.exists(f"screening/{exp_name}_{epoch}"):
        os.mkdir(f"screening/{exp_name}_{epoch}")
    for j in range(10):
        i_st = 10 * j
        i_end = 10 * (j + 1)
        command = ""
        minmax = []
        for i in range(i_st, i_end):
            if os.path.exists(f"screening/{exp_name}_{epoch}/result_{i}"):
                continue
            minmax.append(i)
            test_result_filename = "./screening/{exp_name}_{epoch}/result_{i}"
            restart_file = "../results/{exp_name}/save_{epoch}.pt"
            command += f"OMP_NUM_THREADS={ncpu} python -u ../test.py " + \
                f"--num_workers=0 --restart_file={restart_file} " + \
                f"--n_gnn=3 --dim_gnn=128 --batch_size=8 " + \
                f"--test_result_filename= {test_result_filename}" + \
                f"--ngpu=1 --interaction_net --potential=\"harmonic\" " + \
                f"--data_dir={data_path}/data " + \
                f"--filename={data_path}/pdb_to_affinity.txt " + \
                f"--key_dir={data_path}/keys/{i}/ " + \
                f">./screening/{exp_name}_{epoch}/test_{i} 2>/dev/null\n"
        if command == "":
            continue
        # print(f"Experiment: {exp_name}, Epoch: {epoch}, Number: {minmax}")
        lines = f"""#!/bin/bash
#PBS -N mseok_screening_test_{exp_name}_{epoch}_{j}
#PBS -l nodes=1:ppn={ncpu}
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

date
source activate pytorch-1.5.0
export OMP_NUM_THREADS=1
{command}"""
        
        print(command)
        continue
        with open(f"jobscript_screening.x", "w") as w:
            w.write(lines)
        os.system(f"qsub jobscript_screening.x")
        time.sleep(30)
