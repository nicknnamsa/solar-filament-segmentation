"""Train EdgeAttNet (semantic segmentation) on the filament training split.

A parallel, separate pipeline from train.py (Mask R-CNN / instance segmentation), since EdgeAttNet's model
interface, loss, and data format are all different -- but it reuses the same PQ-scoring infrastructure
(fast_pq.py) via edgeattnet_predict.predict_instances_edgeattnet, which returns the same (score, rle) format as
predict.predict_instances, so results are directly comparable to V1-V5's numbers.

    python scripts/train_edgeattnet.py --config EdgeAttNet/configs/edgeattnet.yaml

Config keys under `train` (same augment/sample_one_annotator/agreement semantics as dataset.FilamentDataset):
    lr, epochs, batch_size, num_workers, seed
    augment, sample_one_annotator, agreement
    eval_every, eval_cutoffs   -- score PQ on validation every N epochs; best kept as best_pq.pt
    save_every
Config keys under `predict`: mask_threshold (binarizes the probability map before connected components),
score_threshold (default cutoff for keeping a component, same role as in the other configs), min_area.
"""
import argparse
import csv
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_image_uint8
from edgeattnet import EdgeAttNet, bce_dice_loss
from edgeattnet_data import FilamentSemanticDataset, collate_fn
from edgeattnet_predict import predict_instances_edgeattnet
from fast_pq import build_ground_truth, pq_from_stats, sweep_stats
from model import get_device

DEFAULT_CUTOFFS = [round(c, 2) for c in np.arange(0.3, 0.96, 0.05)]


def load_image_1ch(image_dir, file_name):
    """Same convention as edgeattnet_data: single-channel grayscale float in [0, 1]."""
    return load_image_uint8(image_dir, file_name).float() / 255.0


def evaluate_pq(model, cfg, device, val_ds, gt, cutoffs):
    """One prediction per FILE (not per annotator-entry) -- the model doesn't know which annotator's version of
    an image it's scored against; ground truth (`gt`) already holds every annotator's masks per file."""
    model.eval()
    instances = {}
    for file_name in val_ds.files:
        image = load_image_1ch(cfg["data"]["train_images"], file_name)
        instances[file_name] = predict_instances_edgeattnet(
            model, image, device, prob_threshold=cfg["predict"]["mask_threshold"],
            min_area=cfg["predict"].get("min_area", 20))
    model.train()
    stats = sweep_stats(instances, gt, cutoffs)
    return pq_from_stats(stats), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="EdgeAttNet/configs/edgeattnet.yaml")
    parser.add_argument("--limit", type=int, help="use only N train images (and N//2 val) for a quick smoke test")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d, t = cfg["data"], cfg["train"]
    epochs = args.epochs or t["epochs"]
    device = get_device()
    print(f"device: {device}")

    if t.get("seed") is not None:
        random.seed(t["seed"])
        np.random.seed(t["seed"])
        torch.manual_seed(t["seed"])

    def read_split(path, limit=None):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()][:limit]

    train_ids = read_split(d["train_split"], args.limit)
    val_ids = read_split(d["val_split"], args.limit and max(args.limit // 2, 1))
    resize = t.get("resize")   # local-testing / fallback knob; unset for the real, native-2048px run
    train_ds = FilamentSemanticDataset(d["train_images"], d["annotations"], train_ids, augment=t.get("augment"),
                                       sample_one_annotator=t.get("sample_one_annotator", False),
                                       agreement=t.get("agreement"), resize=resize)
    val_ds = FilamentSemanticDataset(d["train_images"], d["annotations"], val_ids, resize=resize)  # never altered
    if resize and t.get("eval_every", 0):
        raise NotImplementedError("PQ evaluation with `resize` set is not implemented (predictions would need "
                                  "upsampling back to native resolution first); set eval_every: 0 for resize tests")
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"],
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"],
                            collate_fn=collate_fn)
    print(f"{len(train_ds)} training items per epoch, {len(val_ds)} validation entries "
          f"({len(val_ds.files)} validation images)")

    eval_every = t.get("eval_every", 0)
    cutoffs = t.get("eval_cutoffs", DEFAULT_CUTOFFS)
    gt = build_ground_truth(COCO(d["annotations"]), val_ds.image_ids) if eval_every else None

    model = EdgeAttNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(t["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, ckpt_dir / "config.yaml")
    history_path = ckpt_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "pq", "best_cutoff", "minutes"])

    best_pq = -1.0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                val_loss += bce_dice_loss(model(images), masks).item()
        val_loss /= max(len(val_loader), 1)

        pq_text, pq_value, pq_cutoff = "", "", ""
        if eval_every and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs):
            pq, _ = evaluate_pq(model, cfg, device, val_ds, gt, cutoffs)
            b = int(pq.argmax())
            pq_value, pq_cutoff = round(float(pq[b]), 4), cutoffs[b]
            pq_text = f" | PQ {100 * pq[b]:.1f}% at cutoff {cutoffs[b]}"
            print("PQ by cutoff: " + " ".join(f"{c}:{100 * p:.1f}" for c, p in zip(cutoffs, pq)), flush=True)
            if pq[b] > best_pq:
                best_pq = pq[b]
                torch.save(model.state_dict(), ckpt_dir / "best_pq.pt")
                pq_text += "  (new best)"

        minutes = (time.time() - epoch_start) / 60
        peak = f" peak_gpu_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if device.type == "cuda" else ""
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={minutes:.1f}min{peak}"
              f"{pq_text}", flush=True)
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, round(train_loss, 4), round(val_loss, 4), pq_value, pq_cutoff,
                                    round(minutes, 2)])

        torch.save(model.state_dict(), ckpt_dir / "last.pt")
        if t.get("save_every") and (epoch + 1) % t["save_every"] == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
