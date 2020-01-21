import torch
import torch.nn as nn

def load_data(filename):
    data = dict()
    with open(filename) as f:
        lines = f.readlines()[1:]
        lines = [l.strip() for l in lines]
        for l in lines:
            l = l.split(',')
            if len(l)==5: l.append(-100.0)
            k = l[1]
            if k not in data.keys(): data[k] = []
            data[k].append((int(l[2]),int(l[3]),l[4],float(l[5])))

    return data
        
def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    import random
    empty = []
    if ngpus>0:
        fn = f'empty_gpu_check_{random.choice([i for i in range(10000)])}'
        for i in range(4):
            os.system(f'nvidia-smi -i {i} | grep "No running" | wc -l > {fn}')
            with open(fn) as f:
                out = int(f.read())
            if int(out)==1:
                empty.append(i)
            if len(empty)==ngpus: break
        if len(empty)<ngpus:
            print ('avaliable gpus are less than required', len(empty), ngpus)
            exit(-1)
        os.system(f'rm -f {fn}')        
    
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','

    return cmd

def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        if device.type=='cpu':
            model.load_state_dict(torch.load(load_save_file, map_location='cpu')) 
        else:
            model.load_state_dict(torch.load(load_save_file)) 
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    return model
