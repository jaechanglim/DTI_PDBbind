#!/bin/bash
python -u ./train.py \
--batch_size=8 \
--save_dir=save \
--tensorboard_dir=run \
--n_gnn=3 \
--dim_gnn=128 \
--ngpu=1 \
--train_result_filename=result_train.txt \
--test_result_filename=result_test.txt \
--train_result_docking_filename=result_docking_train.txt \
--test_result_docking_filename=result_docking_test.txt \
--train_result_screening_filename=result_screening_train.txt \
--test_result_screening_filename=result_screening_test.txt \
--data_dir=./data/pdbbind_v2019_refined/data/ \
--filename=./data/pdbbind_v2019_refined/pdb_to_affinity.txt \
--key_dir=./data/pdbbind_v2019_refined/keys/ \
--data_dir2=./data/pdbbind_v2019_docking_nowater/data/ \
--filename2=./data/pdbbind_v2019_docking_nowater/pdb_to_affinity.txt \
--key_dir2=./data/pdbbind_v2019_docking_nowater/keys/ \
--data_dir3=./data/pdbbind_v2019_random_screening/data/ \
--filename3=./data/pdbbind_v2019_random_screening/pdb_to_affinity.txt \
--key_dir3=./data/pdbbind_v2019_random_screening/keys/ \
--data_dir4=./data/pdbbind_v2019_cross_screening/data/ \
--filename4=./data/pdbbind_v2019_cross_screening/pdb_to_affinity.txt \
--key_dir4=./data/pdbbind_v2019_cross_screening/keys/ \
--num_workers=9 \
--potential='harmonic' \
--num_epochs=1001 \
--dropout_rate=0.1 \
--interaction_net \
--dev_vdw_radius=0.2 \
--save_every=100 > output
