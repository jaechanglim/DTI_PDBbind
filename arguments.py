import argparse

def parser(command):
    arg_command = command[1:]
    home = '/home/share/DTI_PDBbind2/data_pdbbind2'
    parser = argparse.ArgumentParser(description='parser for train and test')
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
                        default=home+'/pdb_to_affinity.txt')
    parser.add_argument('--key_dir',
                        help='key directory',
                        type=str,
                        default='/home/udg/msh/urp/DTI_PDBbind/keys')
    parser.add_argument('--data_dir',
                        help='data file path',
                        type=str,
                        default=home+'/data/')
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

    # for train
    if "train.py" in command[0]:
        parser.add_argument('--lr',
                            help="learning rate",
                            type=float, 
                            default=1e-4)
        parser.add_argument("--lr_decay", 
                            help="learning rate decay", 
                            type=float, 
                            default=1.0)
        parser.add_argument("--weight_decay", 
                            help="weight decay", 
                            type=float, 
                            default=0.0)
        parser.add_argument('--num_epochs', 
                            help='number of epochs', 
                            type=int, 
                            default=100)
        parser.add_argument('--train_output_filename',
                            help='train output filename',
                            type=str,
                            default='train.txt')
        parser.add_argument('--eval_output_filename',
                            help='evaluation output filename',
                            type=str,
                            default='eval.txt')
        parser.add_argument("--dropout_rate",
                            help="dropout rate",
                            type=float,
                            default=0.0)
        parser.add_argument("--loss2_ratio",
                            help="loss2 ratio",
                            type=float,
                            default=1.0)
        parser.add_argument("--save_dir",
                            help='save directory of model save files',
                            type=str)
        parser.add_argument("--tensorboard_dir",
                            help='save directory of tensorboard log files',
                            type=str)

    # for test
    if "test.py" in command[0]:
        parser.add_argument('--test_output_filename',
                            help='test output filename',
                            type=str,
                            default='test.txt')

    # for multi test
    if "multi_test" in command[0]:
        parser.add_argument('--epoch_interval',
                            help='epoch interval for test',
                            type=int,
                            default=0)

    args = parser.parse_args(arg_command)
    print (args)
    return args
