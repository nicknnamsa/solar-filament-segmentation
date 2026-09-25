"""Run inference and write an RLE submission CSV (columns: filament_id, segmentation_rle).

    python scripts/predict.py --checkpoint V2/checkpoints/run1/best_pq.pt --config V2/configs/maskrcnn_v2.yaml
    python scripts/predict.py --checkpoint ... --split val      # predictions for the local validation images
    python scripts/predict.py --checkpoint ... --tta            # test-time augmentation: pool 8 dihedral views (slower)

Instances are made non-overlapping (higher-scoring instances keep contested pixels), and ids are "<image>_<k>".
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_util
from scipy import ndimage as ndi

from dataset import load_image_tensor
from model import build_model, get_device


def mask_to_rle(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


@torch.no_grad()
def predict_instances(model, image, device, min_score=0.05, mask_threshold=0.5):
    """Returns [(score, rle_string), ...] sorted by descending score, with no overlapping pixels."""
    out = model([image.to(device)])[0]
    taken = np.zeros(image.shape[1:], dtype=bool)
    instances = []
    for i in torch.argsort(out["scores"], descending=True):
        score = out["scores"][i].item()
        if score < min_score:
            break
        mask = (out["masks"][i, 0] > mask_threshold).cpu().numpy() & ~taken
        if mask.any():
            taken |= mask
            instances.append((score, mask_to_rle(mask)))
    return instances


def merge_nearby_instances(instances, size, radius):
    """Groups instances whose masks, after independently dilating each by `radius` pixels, touch or overlap --
    merging each group into ONE instance: the union of the ORIGINAL (undilated) mask pixels, so the submitted
    shape never grows, only identities merge (same principle as edgeattnet_predict.py's merge_radius). Scored by
    the group's highest member score. radius<=0 is a no-op.

    Rationale (see V5/README.md): Mask R-CNN sometimes emits multiple adjacent/overlapping-but-not-quite-matching
    detections for what should be one filament. Measured on the validation set (V5, cutoff 0.825): radius=1
    reduces false positives 356->287 at the cost of a few marginal true positives (716->707), net PQ 40.9%->41.7%,
    stable from radius 1 through at least 12 (no over-merging observed in that range).

    `instances` should already be filtered to the score threshold you intend to submit at -- this was measured
    with merging applied AFTER thresholding, not before, and that order matters (merging low-score noise in
    changes which groups form).
    """
    if radius <= 0 or not instances:
        return instances
    scores = [s for s, _ in instances]
    masks = [mask_util.decode({"size": list(size), "counts": r.encode("ascii")}).astype(bool) for _, r in instances]
    n = len(masks)
    dilated = [ndi.binary_dilation(m, iterations=radius) for m in masks]

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if (dilated[i] & dilated[j]).any():
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out = []
    for idx in groups.values():
        union_mask = np.zeros(size, dtype=bool)
        for i in idx:
            union_mask |= masks[i]
        out.append((max(scores[i] for i in idx), mask_to_rle(union_mask)))
    return sorted(out, key=lambda x: -x[0])


def load_model(cfg, checkpoint, device):
    cfg["model"]["pretrained"] = False          # weights come from the checkpoint
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def instances_to_df(instances_by_stem, score_threshold):
    rows = []
    for stem, instances in instances_by_stem.items():
        kept = [rle for score, rle in instances if score >= score_threshold]
        rows += [(f"{stem}_{k}", rle) for k, rle in enumerate(kept)]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="V2/configs/maskrcnn_v2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--score-threshold", type=float, help="default: predict.score_threshold from the config")
    parser.add_argument("--out", help="output CSV (default: outputs/submissions/<checkpoint dir>_<split>.csv)")
    parser.add_argument("--tta", action="store_true", help="pool predictions from 8 dihedral views (~8x slower); see tta.py")
    parser.add_argument("--cluster-iou", type=float, default=0.3, help="only used with --tta")
    parser.add_argument("--merge-radius", type=int, default=0,
                        help="merge nearby/overlapping instances after thresholding (see merge_nearby_instances); "
                             "measured +0.8pp PQ on V5 at radius=1, stable through radius=12 -- see V5/README.md")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p = cfg["predict"]
    thr = args.score_threshold if args.score_threshold is not None else p["score_threshold"]
    device = get_device()
    model = load_model(cfg, args.checkpoint, device)

    if args.tta:
        from tta import predict_instances_tta
        run_predict = lambda image: predict_instances_tta(model, image, device, mask_threshold=p["mask_threshold"],
                                                           cluster_iou=args.cluster_iou)
    else:
        run_predict = lambda image: predict_instances(model, image, device, mask_threshold=p["mask_threshold"])

    if args.split == "test":
        image_dir = Path(cfg["data"]["test_images"])
        files = sorted(f.name for f in image_dir.glob("*") if f.suffix.lower() in (".jpeg", ".jpg", ".png"))
    else:
        import json
        from dataset import read_split
        image_dir = Path(cfg["data"]["train_images"])
        with open(cfg["data"]["annotations"]) as f:
            file_of = {im["id"]: im["file_name"] for im in json.load(f)["images"]}
        files = sorted({file_of[i] for i in read_split(cfg["data"]["val_split"])})

    instances = {}
    for n, name in enumerate(files, 1):
        image = load_image_tensor(image_dir, name)
        preds = run_predict(image)
        if args.merge_radius:
            # merging is only validated AFTER thresholding -- see merge_nearby_instances' docstring
            preds = merge_nearby_instances([(s, r) for s, r in preds if s >= thr], image.shape[1:], args.merge_radius)
        instances[Path(name).stem] = preds
        print(f"\r{n}/{len(files)}", end="", flush=True)
    print()

    thr_for_df = -1.0 if args.merge_radius else thr   # already thresholded before merging in that case
    df = instances_to_df(instances, thr_for_df)
    suffix = "_tta" if args.tta else ""
    suffix += f"_merge{args.merge_radius}" if args.merge_radius else ""
    out = Path(args.out or Path(p["output_dir"]) / f"{Path(args.checkpoint).parent.name}_{args.split}{suffix}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    empty = sum(1 for s in instances if not any(sc >= thr_for_df for sc, _ in instances[s]))
    print(f"wrote {len(df)} filaments over {len(files)} images to {out} ({empty} images with no predictions)")


if __name__ == "__main__":
    main()
