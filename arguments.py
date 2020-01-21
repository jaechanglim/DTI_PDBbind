import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        help='batch size',
                        type=int,
                        default=1)
    parser.add_argument('--num_workers',
                        help='number of workers',
                        type=int,
                        default=7) 
    parser.add_argument('--dim_gnn',
                        help='dim_gnn',
                        type=int,
                        default=32) 
    parser.add_argument("--n_gnn",
                        help="depth of gnn layer",
                        type=int,
                        default=3)
    parser.add_argument('--ngpu',
                        help='ngpu',
                        type=int,
                        default=1) 
    parser.add_argument('--restart_file',
                        help='restart file',
                        type=str) 
    parser.add_argument('--filename',
                        help='filename',
                        type = str,
                        default='/home/share/DTI_PDBbind/data_pdbbind/\
                                pdb_to_affinity.txt')
    parser.add_argument('--exp_name',
                        help='experiment name',
                        type=str)
    parser.add_argument('--test_output_filename',
                        help='test output filename',
                        type=str,
                        default='test.txt')
    parser.add_argument('--key_dir',
                        help='key directory',
                        type=str,
                        default='/home/udg/msh/urp/DTI_PDBbind/keys')
    parser.add_argument('--data_dir',
                        help='data file path',
                        type=str,
                        default='/home/share/DTI_PDBbind/data_pdbbind/data/')
    parser.add_argument("--filter_spacing",
                        help="filter spacing",
                        type=float,
                        default=0.1)
    parser.add_argument("--filter_gamma",
                        help="filter gamma",
                        type=float,
                        default=10)
    parser.add_argument("--potential",
                        help="potential",
                        type=str, 
                        default='morse_all_pair', 
                        choices=['morse',
                                 'harmonic',
                                 'morse_all_pair',
                                 'harmonic_interaction_specified'])
    parser.add_argument('--epoch_interval',
                        help='epoch interval for test',
                        type=int,
                        default=0)
    args = parser.parse_args()
    print (args)
    return args
