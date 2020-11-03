import glob

total = glob.glob("./data/????")
total = [t.split("/")[-1] for t in total]
with open("../power_scoring/CoreSet.dat", "r") as f:
    lines = f.readlines()[1:]
    lines = [line.split() for line in lines]
    lines = [[line[0], line[3]] for line in lines]
    dic = dict(lines)
with open("./pdb_to_affinity.txt", "w") as f:
    for t in total:
        val = dic[t]
        f.write(f"{t}\t{val}\n")

