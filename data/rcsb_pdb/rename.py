import glob
import os


def select(fn):
    # fn = './5c5t.sdf'
    directory = '/'.join(fn.split('/')[:-1])
    if not len(os.listdir(directory)):
        print(directory)
        os.removedirs(directory)
    key = fn.split('/')[-2]
    backup_fn = f'{key}_old.sdf'
    new_fn = f'{directory}/{key}.sdf'
    backup = f'{directory}/{backup_fn}'
    os.system(f'mv {fn} {backup}')
    upperkey = key.upper()
    with open(backup, 'r') as r:
        lines = r.readlines()
        new_lines = []
        here = False
        for l in lines:
            if upperkey in l:
                here = True
            if here:
                new_lines.append(l)
            if '$$$$' in l:
                here = False

    with open(new_fn, 'w') as w:
        for nl in new_lines:
            w.write(nl)

total = glob.glob('./refined_data/????/*.sdf')
for t in total[1:]:
    select(t)
