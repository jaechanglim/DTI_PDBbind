import sys
import os
import glob
import multiprocessing
from multiprocessing import Pool


def split_pdbqt(pdbqt):
    prefix = pdbqt.split("/")[-1].split(".")[0]
    direc = "/".join(pdbqt.split("/")[:-2])
    if not os.path.exists(direc + "/result_split_pdbqt/"):
        os.mkdir(direc + "/result_split_pdbqt")
    autodock_path = "/home/mseok/programs/autodock_vina/autodock_vina_1_1_2_linux_x86/bin/vina_split"
    command = f"{autodock_path} \
            --input {pdbqt} \
            --ligand {direc}/result_split_pdbqt/{prefix}_"
    os.system(command)


def cut_pdbqt(pdbqt):
    if os.path.exists(pdbqt):
        direc = "/".join(pdbqt.split("/")[:-2])
        pdb = direc + "/result_pdb/" + pdbqt[:-5].split("/")[-1] + "pdb"
        os.system(f"cut -c 1-60,70-79 {pdbqt} > {pdb}")


if __name__ == "__main__":
    start, end = int(sys.argv[1]), int(sys.argv[2])
    pool = Pool(4)
    names1 = glob.glob("./result_pdbqt/*.pdbqt")
    names1_list = [n.split("/")[-1].split(".")[0] for n in names1[start:end]]
    r = pool.map_async(split_pdbqt, names1[start:end])
    r.wait()
    pool.close()
    pool.join()
    print("split_pdbqt_done")

    pool = Pool(4)
    names2 = [f"./result_split_pdbqt/{name1}_{i}.pdbqt" for name1 in names1_list for i in range(5)]
    r = pool.map_async(cut_pdbqt, names2)
    r.wait()
    pool.close()
    pool.join()
    print("converting pdbqt to pdb done")
