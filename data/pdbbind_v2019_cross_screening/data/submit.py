import os
import glob
import sys


total = glob.glob("../result_pdb/*_out_1.pdb")
length = len(total)

n_proc = int(sys.argv[1])
for i in range(n_proc):
    n = length // n_proc
    start = i * n
    end = (i + 1) * n if i != n_proc-1 else -1
    print(start, end)
    command = f"""#!/bin/bash
#PBS -N msh_cross_pp_final_{i}
#PBS -l nodes=1:ppn=4
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`
date

source ~/.bashrc
source activate mseok
export OMP_NUM_THREADS=1

python ./preprocess.py {start} {end} 1>pp{i}.out 2>/dev/null
data"""
    with open("./jobscript.x", "w") as w:
        w.write(command)
    os.system("qsub jobscript.x")
