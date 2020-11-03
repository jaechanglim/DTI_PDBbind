import os
import time

for i in range(6, 10):
    command = f"""#!/bin/bash
#PBS -N msh_DTI_check{i}
#PBS -l nodes=1:ppn=7
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`
date

source ~/.bashrc
conda activate mseok
export OMP_NUM_THREADS=1

number={i}
exp_name=check$number

python -u ./train.py \
--batch_size=8 \
--save_dir=save_$number.0 \
--tensorboard_dir=run_$number.0 \
--n_gnn=3 \
--dim_gnn=128 \
--ngpu=1 \
--train_result_filename=./output/"$exp_name"_train.txt \
--test_result_filename=./output/"$exp_name"_test.txt \
--train_result_docking_filename=./output/"$exp_name"_docking_train.txt \
--test_result_docking_filename=./output/"$exp_name"_docking_test.txt \
--train_result_screening_filename=./output/"$exp_name"_screening_train.txt \
--test_result_screening_filename=./output/"$exp_name"_screening_test.txt \
--loss_der1_ratio=10.0 \
--loss_der2_ratio=10.0 \
--min_loss_der2=-20.0 \
--loss_docking_ratio=10.0 \
--loss_screening_ratio=5.0 \
--loss_screening2_ratio=5.0 \
--data_dir=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_refined/data \
--filename=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_refined/pdb_to_affinity.txt \
--key_dir=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_refined/ \
--data_dir2=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_docking_nowater/data/ \
--filename2=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_docking_nowater/pdb_to_affinity.txt \
--key_dir2=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_docking_nowater/ \
--data_dir3=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_random_screening/data/ \
--filename3=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_random_screening/pdb_to_affinity.txt \
--key_dir3=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_random_screening/ \
--data_dir4=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_cross_screening/data/ \
--filename4=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_cross_screening/pdb_to_affinity.txt \
--key_dir4=/home/wykgroup/mseok/data/DTI_PDBbind/pdbbind_v2019_cross_screening/ \
--num_workers=9 \
--potential="harmonic" \
--num_epochs=1001 \
--dropout_rate=0.1 \
--interaction_net \
--dev_vdw_radius=0.2 \
--save_every=100 \
> output_$number.0"""
    with open("./jobscript.x", "w") as w:
        w.write(command)
    os.system("qsub jobscript.x")
    time.sleep(10)
