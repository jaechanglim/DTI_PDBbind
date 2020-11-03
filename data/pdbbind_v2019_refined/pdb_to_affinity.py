import glob


total = glob.glob("./data/????")
total = [t.split("/")[-1] for t in total]
with open("../index_pdbbind_v2019/index/INDEX_refined_data.2019", "r") as f:
    lines = f.readlines()[6:]
    lines = [line.split() for line in lines]
    lines = [[line[0], line[3]] for line in lines]
    dic = dict(lines)
with open("./pdb_to_affinity.txt", "w") as f:
    for k, v in dic.items():
        f.write(f"{k}\t{v}\n")
