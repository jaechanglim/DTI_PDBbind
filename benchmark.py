import os

ngpu=1
ncpu=1

#for t in ['casf2013_screening_power']:
for t in ['csar1', 'csar2', 'pdbbind', 'casf2013_scoring_power', \
        'casf2013_docking_power']:
#for t in ['casf2013_scoring_power', \
#        'casf2013_docking_power']:
#for t in ['casf2013_docking_power']:
    for p in ['harmonic']:
    #for p in ['harmonic', 'morse_all_pair']:
        #for epoch in []:
        for epoch in [i*10 for i in range(15,31)]:
            print (t, p, epoch)
            command = f'OMP_NUM_THREADS={ncpu} python -u ../test.py --batch_size=64 '+\
                      f'--num_workers=0 --restart_file=save_{p}/save_{epoch}.pt '+\
                      f'--n_gnn=3 --dim_gnn=128 '+\
                      f'--test_output_filename=result_{t}_{p}_{epoch} '+\
                      f'--ngpu={ngpu} --potential=\'{p}\' '
            if t=='csar2':
                command+=f'--data_dir=../../data_csar_nrc_hiq_set2/data '+\
                         f'--filename=../../data_csar_nrc_hiq_set2/pdb_to_affinity.txt '+\
                         f'--key_dir=../keys_csar2/'
            elif t=='csar1':
                command+=f'--data_dir=../../data_csar_nrc_hiq_set1/data '+\
                         f'--filename=../../data_csar_nrc_hiq_set1/pdb_to_affinity.txt '+\
                         f'--key_dir=../keys_csar1/ '
            elif t=='pdbbind':
                command+=f'--data_dir=../../data_pdbbind2/data '+\
                         f'--filename=../../data_pdbbind2/pdb_to_affinity.txt '+\
                         f'--key_dir=../keys/ '
            elif t=='casf2013_scoring_power':
                command+=f'--data_dir=../../data_casf2013_coreset/data '+\
                         f'--filename=../../data_casf2013_coreset/pdb_to_affinity.txt '+\
                         f'--key_dir=../keys_casf2013_scoring_power/ '
            elif t=='casf2013_docking_power':
                command+=f'--data_dir=../../data_casf2013_decoy_docking/data '+\
                         f'--filename=../../data_casf2013_decoy_docking/pdb_to_affinity.txt '+\
                         f'--key_dir=../keys_casf2013_docking_power/ '
            elif t=='casf2013_screening_power':
                command+=f'--data_dir=../../data_casf2013_decoy_screening/data '+\
                         f'--filename=../../data_casf2013_decoy_screening/pdb_to_affinity.txt '+\
                         f'--key_dir=../keys_casf2013_screening_power/ '
                                     
            command+=f' > test_{t}_{p}_{epoch}'
            os.system(command)
