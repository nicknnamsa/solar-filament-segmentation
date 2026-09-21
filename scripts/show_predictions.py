"""Compare the model's predictions on a held-out validation image with each annotator's masks.

Output is one PNG: the raw image, one panel per annotator, and the model's prediction. Model filaments are green
if they overlap some annotator filament with IoU > 0.5 (would count as a match), and red if they match none.
Panel titles give the per-annotator PQ of the model's prediction.

    python scripts/show_predictions.py --checkpoint outputs/checkpoints/raw/last.pt
    python scripts/show_predictions.py --checkpoint outputs/checkpoints/raw/last.pt --file 20140609195854Bh.jpeg
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from pycocotools import mask as mask_util

from dataset import load_image_tensor, read_split
from metrics import get_overlap_df, pq_breakdown, polygon_to_rle
from model import get_device
from predict import load_model, predict_instances

PANEL = 1024
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]
GREEN, RED = (60, 220, 90), (255, 60, 60)


def stretch(gray):
    lo, hi = np.percentile(gray, [1, 99.5])
    return np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay(base, masks, colors):
    """Blend full-size binary masks onto a PANEL-sized RGB base; also outline each mask."""
    step = masks[0].shape[0] // PANEL if masks else 1
    out = np.asarray(base, dtype=np.float32).copy()
    for m, c in zip(masks, colors):
        small = m[::step, ::step][:PANEL, :PANEL].astype(bool)
        edge = small & ~(np.roll(small, 1, 0) & np.roll(small, -1, 0) & np.roll(small, 1, 1) & np.roll(small, -1, 1))
        out[small] = 0.6 * out[small] + 0.4 * np.array(c)
        out[edge] = c
    return Image.fromarray(out.astype(np.uint8))


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, PANEL, 22], fill=(0, 0, 0))
    d.text((8, 5), text, fill=(255, 255, 255))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maskrcnn_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--file", help="a validation image file name; default: a random validation image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoff", type=float, default=0.8, help="score cutoff for keeping a predicted filament")
    parser.add_argument("--out", default="outputs/visualizations")
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
    by_file = {}
    for i in val_ids:
        by_file.setdefault(file_of[i], []).append(i)
    if args.file:
        assert args.file in by_file, f"{args.file} is not in the validation split"
        name = args.file
    else:
        multi = sorted(f for f, ids in by_file.items() if len(ids) >= 2)
        name = random.Random(args.seed).choice(multi)
    ids = sorted(by_file[name])
    print(f"validation image: {name}  ({len(ids)} annotators)")

    device = get_device()
    model = load_model(cfg, args.checkpoint, device)
    image = load_image_tensor(d["train_images"], name)
    preds = [(s, r) for s, r in predict_instances(model, image, device, mask_threshold=cfg["predict"]["mask_threshold"])
             if s >= args.cutoff]
    h, w = size_of[ids[0]]
    dec = lambda rles: [mask_util.decode({"size": [h, w], "counts": r.encode("ascii")}) for r in rles]
    pred_rles = [r for _, r in preds]
    pred_masks = dec(pred_rles)

    gt_rles = {i: [polygon_to_rle(a["segmentation"], h, w) for a in anns.get(i, [])] for i in ids}
    gt_masks = {i: dec(gt_rles[i]) for i in ids}

    # score the model against each annotator with the official code
    cols = ["filament_id", "segmentation_rle"]
    stem = Path(name).stem
    pred_df = pd.DataFrame([(f"{stem}_{k}", r) for k, r in enumerate(pred_rles)], columns=cols)
    per_annotator = {}
    for i in ids:
        gt_df = pd.DataFrame([(f"{i}_{k}", r) for k, r in enumerate(gt_rles[i])], columns=cols)
        per_annotator[i] = pq_breakdown(get_overlap_df(gt_df, pred_df))

    # which predictions would count as a match for at least one annotator?
    all_gt = [r for i in ids for r in gt_rles[i]]
    to_rle = lambda rs: [{"size": [h, w], "counts": r.encode("ascii")} for r in rs]
    if pred_rles and all_gt:
        best_iou = mask_util.iou(to_rle(pred_rles), to_rle(all_gt), [0] * len(all_gt)).max(axis=1)
    else:
        best_iou = np.zeros(len(pred_rles))

    base_gray = stretch(np.asarray(Image.open(Path(d["train_images"]) / name).convert("L"), dtype=np.float32))
    base = Image.fromarray((base_gray * 255).astype(np.uint8)).resize((PANEL, PANEL), Image.LANCZOS).convert("RGB")

    panels = [label(base.copy(), f"{name}  (raw image, held-out validation)")]
    for i in ids:
        ms = gt_masks[i]
        panels.append(label(overlay(base, ms, [PALETTE[k % len(PALETTE)] for k in range(len(ms))]),
                            f"annotator {i.split('-')[0]}: {len(ms)} filaments"))
    n_match = int((best_iou > 0.5).sum())
    pred_colors = [GREEN if v > 0.5 else RED for v in best_iou]
    pq_text = ", ".join(f"{100 * per_annotator[i]['pq']:.0f}%" for i in ids)
    panels.append(label(overlay(base, pred_masks, pred_colors),
                        f"MODEL: {len(preds)} filaments, {n_match} green (match an annotator) / {len(preds) - n_match} red. "
                        f"PQ vs each annotator: {pq_text}"))

    ncols = 3
    nrows = -(-len(panels) // ncols)
    sheet = Image.new("RGB", (PANEL * ncols, PANEL * nrows))
    for k, p in enumerate(panels):
        sheet.paste(p, ((k % ncols) * PANEL, (k // ncols) * PANEL))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"prediction_{stem}.png"
    sheet.save(path)
    print(f"model kept {len(preds)} filaments at cutoff {args.cutoff}; {n_match} match at least one annotator")
    for i in ids:
        b = per_annotator[i]
        print(f"  vs annotator {i.split('-')[0]}: PQ {100 * b['pq']:.1f}%  (matched {b['tp']}, spurious {b['fp']}, missed {b['fn']})")
    print(path)


if __name__ == "__main__":
    main()
