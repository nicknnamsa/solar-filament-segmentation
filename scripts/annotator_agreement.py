"""How well do human annotators agree with each other, under the competition's own metric?

Many images were annotated by 2-3 people. This takes one annotator's masks and scores them as if they were
model predictions against the other annotators' masks, using the official Panoptic Quality (PQ) metric.
That gives a rough ceiling: a model cannot be expected to agree with the annotators better than they agree
with each other.

    python scripts/annotator_agreement.py
    python scripts/annotator_agreement.py --split val

"Annotator #1/#2/#3" means the lowest / middle / highest annotator id within each image; it is not the same
person across images.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from metrics import get_overlap_df, get_pq_score, pq_breakdown, polygon_to_rle


def build_dataframes(rank, by_file, anns_by_image, sizes, allowed_files):
    """Annotator number `rank` in each image is the 'prediction'; every other annotator of that image is ground truth."""
    pred_rows, gt_rows = [], []
    for file_name, ids in by_file.items():
        if file_name not in allowed_files or len(ids) < 2 or rank >= len(ids):
            continue
        h, w = sizes[file_name]
        stem = Path(file_name).stem
        for k, a in enumerate(anns_by_image[ids[rank]]):
            pred_rows.append((f"{stem}_{k}", polygon_to_rle(a["segmentation"], h, w)))
        for other in ids[:rank] + ids[rank + 1:]:
            for k, a in enumerate(anns_by_image[other]):
                gt_rows.append((f"{other}_{k}", polygon_to_rle(a["segmentation"], h, w)))
    columns = ["filament_id", "segmentation_rle"]
    return pd.DataFrame(gt_rows, columns=columns), pd.DataFrame(pred_rows, columns=columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="V2/configs/maskrcnn_v2.yaml")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all",
                        help="which images to use (default: all training images)")
    args = parser.parse_args()

    with open(args.config) as f:
        d = yaml.safe_load(f)["data"]
    with open(d["annotations"]) as f:
        coco = json.load(f)

    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)
    by_file, sizes, id_to_file = defaultdict(list), {}, {}
    for im in coco["images"]:
        by_file[im["file_name"]].append(im["id"])
        sizes[im["file_name"]] = (im["height"], im["width"])
        id_to_file[im["id"]] = im["file_name"]
    for ids in by_file.values():
        ids.sort()

    if args.split == "all":
        allowed = set(by_file)
    else:
        with open(d[f"{args.split}_split"]) as f:
            allowed = {id_to_file[line.strip()] for line in f if line.strip()}

    n_files = sum(1 for fn in allowed if len(by_file[fn]) >= 2)
    print(f"{n_files} images with 2+ annotators ({args.split} split). Scoring takes a minute or two...\n")

    all_overlaps = []
    for rank in range(3):
        start = time.time()
        gt_df, pred_df = build_dataframes(rank, by_file, anns_by_image, sizes, allowed)
        if gt_df.empty:
            continue
        overlap_df = get_overlap_df(gt_df, pred_df)
        all_overlaps.append(overlap_df)
        b = pq_breakdown(overlap_df)
        assert abs(b["pq"] - get_pq_score(overlap_df)) < 1e-9   # our breakdown must match the official function
        print(f"Annotator #{rank + 1} as predictions, scored against the other annotators:")
        print(f"    PQ = {100 * b['pq']:.1f}%   (matched {b['tp']}, spurious {b['fp']}, missed {b['fn']}, "
              f"mean IoU of matches {b['mean_iou_of_matches']:.2f})   [{time.time() - start:.0f}s]")

    pooled = pd.concat(all_overlaps, ignore_index=True)
    b = pq_breakdown(pooled)
    print(f"\nAll annotator pairs pooled:  PQ = {100 * b['pq']:.1f}%")
    print("This is roughly the score a model would get by matching a typical human annotator.")


if __name__ == "__main__":
    main()
