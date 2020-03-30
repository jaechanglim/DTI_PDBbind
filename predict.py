import argparse
import utils
import numpy as np
import model 
import os
import torch
import time
import torch.nn as nn
import sys
import glob
import arguments
import dataset
#argument
args = arguments.parser(sys.argv)

#model
cmd = utils.set_cuda_visible_device(args.ngpu)
os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
if args.potential=='morse': model = model.DTILJ(args)
elif args.potential=='morse_all_pair': model = model.DTILJAllPair(args)
elif args.potential=='harmonic': model = model.DTIHarmonic(args)
elif args.potential=='harmonic_interaction_specified': model = model.DTIHarmonicIS(args)
else: 
    print (f'No {args.potential} potential')
    exit(-1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, args.restart_file)
model.eval()
#print (args)
#print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
for lf, pf, of in zip(args.ligand_files, args.protein_files, args.output_files):
    st = time.time()

    #read ligand and protein. Then, convert to rdkit object
    m1 = utils.read_molecule(lf)
    m2 = utils.extract_binding_pocket(m1, pf)

    #preprocess: convert rdkit mol obj to feature
    sample = dataset.mol_to_feature(m1, m1, m2, None, 0.0)
    sample['affinity'] = 0.0
    sample['key'] = 'None'
    sample = dataset.tensor_collate_fn([sample])
    sample = utils.dic_to_device(sample, device)

    #run prediction
    pred, _, _ = model(sample, cal_der_loss=False)
    pred = pred.data.cpu().numpy()[0]
    end = time.time()

    #write
    with open(f'{of}', 'w') as w:
        w.write("\n")
        w.write("                             ..:::::::::::::::.... \n" )
        w.write("                   .::== .#%*  +++++++++++++++++++: =##= .:.. \n")
        w.write("           .::=+++++++: =@@@  +++++++++++++++++++. #@@# .+++++==:. \n" )
        w.write("         .:=+++++++++++:    .=++++++++++++++++++++.    .+++++++++=:            \n" )
        w.write("               ..:==++++++++++++++++++++++++++++++++++++++=::.\n" )
        w.write("                       ..:::====++++++++++++++===:::.. \n" )
        w.write("                             .:+**************\n")
        w.write("                         .:++++=. +++++++++=+++: \n")
        w.write("                  .+++=++++=.     +++++++++= :+++. \n")
        w.write("                  +++++:.         +++++++++=   =+++:=: \n")
        w.write("                   .::.           +++++++++=     :+++++ \n")
        w.write("                                  ++=::::++=      :+++: \n")
        w.write("                                  ++:    ++= \n")
        w.write("                                  ++:    ++= \n" )
        w.write("                             .::::++:    ++=.::: \n")
        w.write("                            .*++++++:    +++++++* \n" )
        w.write("\n")
        #w.write("          .,,**////////((((((((((((((((((((######################(/*..           \n")
        #w.write(",,******///////////////((((((((((((((((((((#####################%%%%%%%%%%%%#(/,.\n")
        #w.write("                  ..,**////((((((((((((((((##########((((/*,..                   \n")
        w.write("\n")
        w.write("\n")
        w.write('#Parameter\n')
        w.write(f'Hbond coeff: {model.vina_hbond_coeff.data.cpu().numpy()[0]:.3f}\n')
        w.write(f'Hydrophobic coeff: {model.vina_hydrophobic_coeff.data.cpu().numpy()[0]:.3f}\n')
        w.write(f'Rotor coeff: {model.rotor_coeff.data.cpu().numpy()[0]:.3f}\n')
        w.write('\n')
        w.write('#Prediction\n')
        w.write(f'Total prediction: {pred.sum():.3f} kcal/mol\n')
        w.write(f'VDW : {pred[0]:.3f} kcal/mol\n')
        w.write(f'Hbond : {pred[1]:.3f} kcal/mol\n')
        w.write(f'Metal : {pred[2]:.3f} kcal/mol\n')
        w.write(f'Hydrophobic : {pred[3]:.3f} kcal/mol\n')
        w.write(f'\nTime : {end-st:.3f} s\n')

