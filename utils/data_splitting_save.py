def save_split(filename, paths, labels):
    with open(filename, "w") as f:
        for p, l in zip(paths, labels):
            f.write(f"{p}\t{l}\n")