#!/bin/bash
python -u ../../test.py \
--batch_size=64 \
--num_workers=0 \
--restart_file=../../../refactored3_5/results/Xhb_check0/save_1000.pt \
--n_gnn=3 \
--dim_gnn=128 \
--dev_vdw_radius=0.0 \
--dropout_rate=0.0 \
--test_result_filename=result_csar1_harmonic_1000 \
--ngpu=0 \
--interaction_net \
--potential="harmonic" \
--data_dir=/home/wykgroup/mseok/data/DTI_PDBbind/jaechang_csar1/data \
--filename=/home/wykgroup/mseok/data/DTI_PDBbind/jaechang_csar1/pdb_to_affinity.txt \
--key_dir=/home/wykgroup/mseok/data/DTI_PDBbind/jaechang_csar1/keys \
> test_csar1_harmonic_1000
