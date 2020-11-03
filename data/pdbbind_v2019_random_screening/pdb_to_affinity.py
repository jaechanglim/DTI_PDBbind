import glob


total = glob.glob("./data/*_????")
total = [t.split("/")[-1] for t in total]
with open("./pdb_to_affinity.txt", "w") as f:
    for t in total:
        f.write(t + "\t5\n")
