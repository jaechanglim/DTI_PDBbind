# DTI_PDBbind

* Train command
```
python -u ../train.py --batch_size=8 \
                   --save_dir=save_1.0 \
                   --tensorboard_dir=run_1.0 \
                   --n_gnn=3 \
                   --dim_gnn=128 \
                   --ngpu=1 \
                   --train_result_filename=result_train_1.0.txt \
                   --test_result_filename=result_test_1.0.txt \
                   --train_result_docking_filename=result_train_docking_1.0.txt \
                   --test_result_docking_filename=result_test_docking_1.0.txt \
                   --train_result_screening_filename=result_train_screening_1.0.txt \
                   --test_result_screening_filename=result_test_screening_1.0.txt \
                   --loss_der1_ratio=10.0 \
                   --loss_der2_ratio=10.0 \
                   --min_loss_der2=-20.0 \
                   --loss_docking_ratio=0.0 \
                   --loss_screening_ratio=1.0 \
                   --data_dir=/home/wykgroup/jaechang/work/ML/PDBbind_DTI/data_pdbbind_v2019/data/ \
                   --filename=/home/wykgroup/jaechang/work/ML/PDBbind_DTI/data_pdbbind_v2019/pdb_to_affinity.txt \
                   --key_dir=../keys_pdbbind_v2019/ \
                   --data_dir2=/home/wykgroup/udg/mseok/data/DTI_PDBbind/docking_data/\
                   --filename2=../keys_pdbbind_v2019_docking/pdb_to_affinity.txt \
                   --key_dir2=../keys_pdbbind_v2019_docking/ \
                   --data_dir3=/home/wykgroup/jaechang/work/ML/PDBbind_DTI/data_pdbbind_random_nowater/data/\
                   --filename3=/home/wykgroup/jaechang/work/ML/PDBbind_DTI/data_pdbbind_random_nowater/pdb_to_affinity.txt \
                   --key_dir3=../keys_pdbbind_random_nowater/ \
                   --num_workers=9 \
                   --potential='harmonic' \
                   --num_epochs=1001 \
                   > output_1.0 2> /dev/null 
```

* Benchmark command
```
python -u ../benchmark.py > output_benchmark 2> /dev/null
```

* Csar1
```
grep 'R:' test_csar1*
```

* Csar2
```
grep 'R:' test_csar2*
```

* Scoring power
```
python ../casf2016_benchmark/scoring_power.py result_casf2016_scoring_power_1.0_0 100
>>> result_casf2016_scoring_power_1.0_0 279 0.114   9.573   [0.03097 ~ 0.21104]
```

* Ranking power
```
python ../casf2016_benchmark/ranking_power.py result_casf2016_scoring_power_1.0_0 100
>>> result_casf2016_scoring_power_1.0_0 -0.068  -0.072  -0.486  [-0.20008 ~ 0.06654]
```

* docking power (not yet implemented)
```
#python ../casf2013_benchmark/docking_power.py 'result_casf2013_docking_power_harmonic_*' result_casf2013_scoring_power_harmonic_0
```
