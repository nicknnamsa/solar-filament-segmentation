"""Run inference and write an RLE submission CSV (columns: filament_id, segmentation_rle).

    python scripts/predict.py --checkpoint outputs/checkpoints/limb/best.pt --config configs/maskrcnn_limb.yaml
    python scripts/predict.py --checkpoint ... --split val      # predictions for the local validation images

Instances are made non-overlapping (higher-scoring instances keep contested pixels), and ids are "<image>_<k>".
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_util

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
    parser.add_argument("--config", default="configs/maskrcnn_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--score-threshold", type=float, help="default: predict.score_threshold from the config")
    parser.add_argument("--out", help="output CSV (default: outputs/submissions/<checkpoint dir>_<split>.csv)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p = cfg["predict"]
    thr = args.score_threshold if args.score_threshold is not None else p["score_threshold"]
    device = get_device()
    model = load_model(cfg, args.checkpoint, device)

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
        instances[Path(name).stem] = predict_instances(model, image, device, mask_threshold=p["mask_threshold"])
        print(f"\r{n}/{len(files)}", end="", flush=True)
    print()

    df = instances_to_df(instances, thr)
    out = Path(args.out or Path(p["output_dir"]) / f"{Path(args.checkpoint).parent.name}_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    empty = sum(1 for s in instances if not any(sc >= thr for sc, _ in instances[s]))
    print(f"wrote {len(df)} filaments over {len(files)} images to {out} ({empty} images with no predictions)")


if __name__ == "__main__":
    main()
