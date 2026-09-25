"""Two free (no-retraining) ways to try to improve on a single model, tested on the validation split:

1. --mode checkpoints: pool predictions from several already-trained checkpoints for the same image.
2. --mode tta:         pool predictions from 8 dihedral transforms (flips/90-degree rotations) of one model.

In both cases, overlapping predictions (IoU >= --cluster-iou) are merged into one cluster. A cluster's mask is its
highest-scoring member's mask, and its "support" is how many distinct sources (checkpoints, or transforms) produced
a member of it. We score PQ two ways: as a normal score cutoff sweep (like evaluate.py), and by requiring a minimum
support count instead of a score, which is the standard way to use agreement across an ensemble or TTA views. The
best result is cross-checked against the official scorer in metrics.py.

    python scripts/tta_ensemble.py --mode checkpoints
    python scripts/tta_ensemble.py --mode tta --config V2/configs/maskrcnn_v2.yaml --checkpoint V2/checkpoints/run1/best_pq.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_util

from dataset import load_image_tensor, read_split
from fast_pq import build_ground_truth, pq_from_stats, sweep_stats
from metrics import get_overlap_df, pq_breakdown
from model import get_device
from predict import instances_to_df, load_model, predict_instances
from tta import SIZE, cluster_instances, dihedral_forward_image, dihedral_inverse_mask

SCORE_CUTOFFS = [round(c, 2) for c in np.arange(0.3, 0.96, 0.05)]


def sweep_by_score(clusters_by_file, cutoff):
    return {f: [(score, rle) for rle, score, _ in cl if score >= cutoff] for f, cl in clusters_by_file.items()}


def sweep_by_support(clusters_by_file, k):
    """Keep clusters with support >= k; give them all the same score so every one is kept at cutoff 0."""
    return {f: [(1.0, rle) for rle, _, support in cl if support >= k] for f, cl in clusters_by_file.items()}


def report(name, instances_by_cutoff_fn, keys, gt, label_fn):
    print(f"\n{name}")
    best = (-1.0, None, None)
    for key in keys:
        instances = instances_by_cutoff_fn(key)
        stats = sweep_stats(instances, gt, [0.0])           # scores already baked in via `key`; just apply matching
        pq = pq_from_stats(stats)[0]
        print(f"  {label_fn(key):<28} PQ {100 * pq:5.1f}%  matched {int(stats[0, 1]):>4}  spurious {int(stats[0, 2]):>4}  "
              f"missed {int(stats[0, 3]):>4}")
        if pq > best[0]:
            best = (pq, key, instances)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["checkpoints", "tta"], required=True)
    parser.add_argument("--config", default="V2/configs/maskrcnn_v2.yaml", help="only used for --mode tta")
    parser.add_argument("--checkpoint", default="V2/checkpoints/run1/best_pq.pt", help="only used for --mode tta")
    parser.add_argument("--cluster-iou", type=float, default=0.3)
    parser.add_argument("--limit", type=int, help="only use the first N validation images (smoke test)")
    args = parser.parse_args()

    with open("V1/configs/maskrcnn_baseline.yaml") as f:
        d = yaml.safe_load(f)["data"]
    with open(d["annotations"]) as f:
        coco = json.load(f)
    file_of = {im["id"]: im["file_name"] for im in coco["images"]}
    val_ids = read_split(d["val_split"])
    files = sorted({file_of[i] for i in val_ids})[: args.limit]
    val_ids = [i for i in val_ids if file_of[i] in files]

    from pycocotools.coco import COCO
    gt = build_ground_truth(COCO(d["annotations"]), val_ids)
    device = get_device()

    if args.mode == "checkpoints":
        sources = [
            ("V1 raw", "V1/configs/maskrcnn_baseline.yaml", "V1/checkpoints/raw/last.pt", "data/raw/train/train_images"),
            ("V1 limb", "V1/configs/maskrcnn_limb.yaml", "V1/checkpoints/limb/last.pt", "data/processed/limb/train_images"),
            ("V2 best_pq", "V2/configs/maskrcnn_v2.yaml", "V2/checkpoints/run1/best_pq.pt", "data/raw/train/train_images"),
        ]
        pooled = {f: [] for f in files}
        for source_id, (name, cfg_path, ckpt, image_dir) in enumerate(sources):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            model = load_model(cfg, ckpt, device)
            t0 = time.time()
            for f in files:
                image = load_image_tensor(image_dir, f)
                for score, rle in predict_instances(model, image, device, mask_threshold=cfg["predict"]["mask_threshold"]):
                    pooled[f].append((score, rle, source_id))
            print(f"[{name}] {len(files)} images in {time.time() - t0:.0f}s", flush=True)
        n_sources = len(sources)

    else:  # tta
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model = load_model(cfg, args.checkpoint, device)
        image_dir = cfg["data"]["train_images"]
        pooled = {f: [] for f in files}
        t0 = time.time()
        for n, f in enumerate(files, 1):
            base = load_image_tensor(image_dir, f)
            for i in range(8):
                transformed = dihedral_forward_image(base, i)
                for score, rle in predict_instances(model, transformed, device, mask_threshold=cfg["predict"]["mask_threshold"]):
                    mask = mask_util.decode({"size": list(SIZE), "counts": rle.encode("ascii")})
                    restored = np.asfortranarray(dihedral_inverse_mask(mask, i))
                    pooled[f].append((score, mask_util.encode(restored)["counts"].decode("ascii"), i))
            if n % 20 == 0 or n == len(files):
                print(f"[TTA] {n}/{len(files)} images, {time.time() - t0:.0f}s elapsed", flush=True)
        n_sources = 8

    print("clustering overlapping predictions...")
    clusters_by_file = {f: cluster_instances(pooled[f], args.cluster_iou) for f in files}

    best_score = report("A) score cutoff on the pooled/clustered predictions (like a single bigger model)",
                        lambda c: sweep_by_score(clusters_by_file, c), SCORE_CUTOFFS, gt, lambda c: f"cutoff {c}")
    best_support = report(f"B) minimum support required (out of {n_sources} sources) instead of a score cutoff",
                          lambda k: sweep_by_support(clusters_by_file, k), list(range(1, n_sources + 1)), gt,
                          lambda k: f"support >= {k}/{n_sources}")

    best = max(best_score, best_support, key=lambda x: x[0])
    cols = ["filament_id", "segmentation_rle"]
    gt_df = pd.DataFrame([(f"{e}_{k}", r) for f, per in gt.items() for e, rl in per.items() for k, r in enumerate(rl)],
                         columns=cols)
    stem_instances = {Path(f).stem: v for f, v in best[2].items()}
    pred_df = instances_to_df(stem_instances, 0.5 if best is best_support else best[1])
    official = pq_breakdown(get_overlap_df(gt_df, pred_df))["pq"]
    print(f"\nbest overall: PQ {100 * best[0]:.1f}%  (cross-check with official scorer: {100 * official:.1f}%, "
          f"{'MATCH' if abs(best[0] - official) < 1e-3 else 'mismatch, treat with caution'})")


if __name__ == "__main__":
    main()
