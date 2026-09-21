"""Score a checkpoint on the local validation split with the competition metric (Panoptic Quality).

Inference runs once at a low score cutoff; then the cutoff is swept, because PQ punishes false positives and the
best cutoff is not obvious in advance. Use the best cutoff as `predict.score_threshold` in the config.

    python scripts/evaluate.py --checkpoint outputs/checkpoints/limb/best.pt --config configs/maskrcnn_limb.yaml
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from dataset import load_image_tensor, read_split
from metrics import get_overlap_df, pq_breakdown, polygon_to_rle
from model import get_device
from predict import instances_to_df, load_model, predict_instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maskrcnn_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--limit", type=int, help="only evaluate N validation images (smoke test)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    with open(d["annotations"]) as f:
        coco = json.load(f)
    file_of = {im["id"]: im["file_name"] for im in coco["images"]}
    size_of = {im["id"]: (im["height"], im["width"]) for im in coco["images"]}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    val_ids = read_split(d["val_split"])
    files = sorted({file_of[i] for i in val_ids})[: args.limit]
    val_ids = [i for i in val_ids if file_of[i] in files]

    # ground truth: every annotator's version of every validation image, ids "<annotator>-<image>_<k>"
    gt_rows = [(f"{i}_{k}", polygon_to_rle(a["segmentation"], *size_of[i]))
               for i in val_ids for k, a in enumerate(anns.get(i, []))]
    gt_df = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])

    device = get_device()
    model = load_model(cfg, args.checkpoint, device)
    instances = {}
    for n, name in enumerate(files, 1):
        image = load_image_tensor(d["train_images"], name)
        instances[Path(name).stem] = predict_instances(model, image, device, mask_threshold=cfg["predict"]["mask_threshold"])
        print(f"\rinference {n}/{len(files)}", end="", flush=True)
    print(f"\n{len(files)} validation images, {len(val_ids)} annotator versions, {len(gt_df)} ground-truth filaments\n")

    print(f"{'cutoff':>7} {'PQ':>7} {'matched':>8} {'spurious':>9} {'missed':>7} {'mean IoU':>9}")
    best = (-1, None)
    for thr in args.thresholds:
        b = pq_breakdown(get_overlap_df(gt_df, instances_to_df(instances, thr)))
        print(f"{thr:>7.2f} {100 * b['pq']:>6.1f}% {b['tp']:>8} {b['fp']:>9} {b['fn']:>7} {b['mean_iou_of_matches']:>9.2f}")
        best = max(best, (b["pq"], thr))
    print(f"\nbest: PQ {100 * best[0]:.1f}% at score cutoff {best[1]}")


if __name__ == "__main__":
    main()
