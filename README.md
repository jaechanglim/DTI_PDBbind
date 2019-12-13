# DTI_PDBbind
'''
python -u train.py --batch_size=8 --num_workers=7 --save_dir=save_0.0 --n_gnn=5 --dim_gnn=128 --ngpu=1 --train_output_filename=train_0.0 --test_output_filename=test_0.0 --loss2_ratio=0.0
'''
'''
python -u test.py --batch_size=8 --num_workers=7 --restart_file=save_0.0/save_2.pt --n_gnn=5 --dim_gnn=128 --ngpu=1 --test_output_filename=test_0.0 --data_dir=../data_pdbbind/data --filename=../data_pdbbind/pdb_to_affinity.txt --key_dir=keys
'''
'''
python -u test.py --batch_size=8 --num_workers=7 --restart_file=save_0.0/save_150.pt --n_gnn=5 --dim_gnn=128 --ngpu=1 --test_output_filename=test_0.0 --data_dir=../data_csar_nrc_hiq_set1/data --filename=../data_csar_nrc_hiq_set1/pdb_to_affinity.txt --key_dir=keys_csar1/
'''
'''
python -u test.py --batch_size=8 --num_workers=7 --restart_file=save_0.0/save_150.pt --n_gnn=5 --dim_gnn=128 --ngpu=1 --test_output_filename=test_0.0 --data_dir=../data_csar_nrc_hiq_set2/data --filename=../data_csar_nrc_hiq_set2/pdb_to_affinity.txt --key_dir=keys_csar2/
'''


