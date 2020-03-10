# DTI_PDBbind

* Train command
```
python -u ../train.py --batch_size=8 \
                   --save_dir=save_harmonic \
                   --n_gnn=3 \
                   --dim_gnn=128 \
                   --ngpu=1 \
                   --train_output_filename=train_harmonic \
                   --test_output_filename=test_harmonic \
                   --loss2_ratio=0.0 \
                   --data_dir=../../data_pdbbind2/data/ \
                   --filename=../../data_pdbbind2/pdb_to_affinity.txt \
                   --num_workers=4 \
                   --potential='harmonic' \
                   --num_epochs=1000 \
                   --key_dir=../keys \
                   > output_harmonic
```

* Benchmark command
```
python -u ../benchmark.py > output_benchmark 2> /dev/null
>>> 0.7
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
grep 'R:' 'test_casf2013_scoring_power_harmonic_*'
```

* Ranking power
```
python ../casf2013_benchmark/ranking_power.py result_casf2013_scoring_power_harmonic_200
```

* docking power
```
python ../casf2013_benchmark/docking_power.py 'result_casf2013_docking_power_harmonic_*' result_casf2013_scoring_power_harmonic_0
```
