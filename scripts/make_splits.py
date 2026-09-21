"""Create train/val image-id splits from the COCO annotation file.

The same image file can appear several times (one entry per annotator), so the split is done per
file: every annotator's version of a file lands on the same side.
"""
import argparse
import json
import random
from collections import defaultdict

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maskrcnn_baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["data"]

    with open(cfg["annotations"]) as f:
        images = json.load(f)["images"]

    by_file = defaultdict(list)
    for im in images:
        by_file[im["file_name"]].append(im["id"])

    files = sorted(by_file)
    random.Random(cfg["seed"]).shuffle(files)
    n_val = max(1, round(len(files) * cfg["val_fraction"]))
    splits = {"val": files[:n_val], "train": files[n_val:]}

    for name, path in (("train", cfg["train_split"]), ("val", cfg["val_split"])):
        ids = sorted(i for fn in splits[name] for i in by_file[fn])
        with open(path, "w") as f:
            f.write("\n".join(ids) + "\n")
        print(f"{name}: {len(splits[name])} files, {len(ids)} annotated entries")


if __name__ == "__main__":
    main()
