"""Fine-tune Mask R-CNN on the filament training split.

    python scripts/train.py --config V2/configs/maskrcnn_v2.yaml

Everything beyond the basic loop is switched on from the `train` section of the config, and is off when the key is
missing, so the configs of the first run (V1/) still train exactly as they did:
    augment, sample_one_annotator, agreement   see dataset.py
    amp                                        mixed precision (CUDA only)
    warmup_iters                               linear learning-rate warmup
    fast_loader                                8-bit images from the loader (converted on the GPU), pinned memory,
                                               persistent workers: keeps a fast GPU fed
    eval_every, eval_cutoffs                   score PQ on the validation images every N epochs and keep the best
                                               epoch by PQ (best_pq.pt, best_pq.json)
    save_every                                 also keep epoch_<n>.pt every N epochs
    mask_loss                                  recall-biased Tversky mask loss instead of plain BCE; see losses.py
    seed
Each run writes history.csv (losses by component, PQ) and a copy of its config next to the checkpoints.
"""
import argparse
import contextlib
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FilamentDataset, apply_jitter, collate_fn, load_image_tensor, read_split
from fast_pq import build_ground_truth, pq_from_stats, sweep_stats
from losses import patch_mask_loss
from model import build_model, get_device
from predict import predict_instances

LOSS_KEYS = ["loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"]
DEFAULT_CUTOFFS = [round(c, 2) for c in np.arange(0.3, 0.96, 0.05)]


def to_device(images, targets, device, jitter=None):
    """Move a batch to the device. 8-bit images (fast_loader) become float 3-channel here; jitter is applied to
    those on the GPU for training batches."""
    out = []
    for image in images:
        image = image.to(device, non_blocking=True)
        if image.dtype == torch.uint8:
            image = image.float().div_(255)
            if image.shape[0] == 1:
                image = image.expand(3, -1, -1)
            if jitter:
                image = apply_jitter(image, jitter)
        out.append(image)
    targets = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
    return out, targets


def short(parts):
    """'cls 0.12 box 0.08 ...' from a dict of mean loss components."""
    names = {"loss_classifier": "cls", "loss_box_reg": "box", "loss_mask": "mask", "loss_objectness": "obj",
             "loss_rpn_box_reg": "rpn"}
    return " ".join(f"{names[k]} {parts[k]:.3f}" for k in LOSS_KEYS)


def evaluate_pq(model, cfg, device, val_ds, gt, cutoffs):
    """Run the model on every validation file once, then score all cutoffs. Returns (pq per cutoff, stats)."""
    model.eval()
    instances = {}
    for file_name in val_ds.files:
        image = load_image_tensor(cfg["data"]["train_images"], file_name)
        instances[file_name] = predict_instances(model, image, device, min_score=min(cutoffs),   # skip detections that no scored cutoff keeps
                                                 mask_threshold=cfg["predict"]["mask_threshold"])
    model.train()
    stats = sweep_stats(instances, gt, cutoffs)
    return pq_from_stats(stats), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="V2/configs/maskrcnn_v2.yaml")
    parser.add_argument("--limit", type=int, help="use only N train images (and N//2 val) for a quick smoke test")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d, t = cfg["data"], cfg["train"]
    epochs = args.epochs or t["epochs"]
    device = get_device()
    print(f"device: {device}")
    patch_mask_loss(cfg)

    if t.get("seed") is not None:
        random.seed(t["seed"])
        np.random.seed(t["seed"])
        torch.manual_seed(t["seed"])

    train_ids = read_split(d["train_split"])[:args.limit]
    val_ids = read_split(d["val_split"])[:args.limit and max(args.limit // 2, 1)]
    fast = bool(t.get("fast_loader", False))
    train_ds = FilamentDataset(d["train_images"], d["annotations"], train_ids, augment=t.get("augment"),
                               sample_one_annotator=t.get("sample_one_annotator", False), agreement=t.get("agreement"),
                               uint8_images=fast)
    val_ds = FilamentDataset(d["train_images"], d["annotations"], val_ids, uint8_images=fast)   # never altered
    extra = dict(pin_memory=device.type == "cuda", persistent_workers=True, prefetch_factor=4) if fast and t["num_workers"] else {}
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=t["batch_size"], shuffle=shuffle,
                                            num_workers=t["num_workers"], collate_fn=collate_fn, **extra)
    jitter = {k: v for k, v in (t.get("augment") or {}).items() if k in ("brightness", "contrast")} if fast else None
    train_loader, val_loader = loader(train_ds, True), loader(val_ds, False)
    print(f"{len(train_ds)} training items per epoch, {len(val_ds)} validation entries "
          f"({len(val_ds.files)} validation images)")

    eval_every = t.get("eval_every", 0)
    cutoffs = t.get("eval_cutoffs", DEFAULT_CUTOFFS)
    gt = build_ground_truth(val_ds.coco, val_ds.image_ids) if eval_every else None

    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    warmup = t.get("warmup_iters", 0)
    print(f"mixed precision: {use_amp}, warmup iterations: {warmup}")

    ckpt_dir = Path(t["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, ckpt_dir / "config.yaml")
    history_path = ckpt_dir / "history.csv"
    history_header = (["epoch", "train_loss", "val_loss"] + [f"train_{k}" for k in LOSS_KEYS]
                      + [f"val_{k}" for k in LOSS_KEYS] + ["pq", "best_cutoff", "minutes"])
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(history_header)

    best_val, best_pq, iteration = float("inf"), -1.0, 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_parts = {k: 0.0 for k in LOSS_KEYS}
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}"):
            if iteration < warmup:                                    # linear warmup from 0.1% of the learning rate
                for g in optimizer.param_groups:
                    g["lr"] = t["lr"] * (0.001 + 0.999 * iteration / warmup)
            elif iteration == warmup and warmup:
                for g in optimizer.param_groups:
                    g["lr"] = scheduler.get_last_lr()[0]
            images, targets = to_device(images, targets, device, jitter)
            with torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext():
                losses = model(images, targets)
                loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for k in LOSS_KEYS:
                train_parts[k] += losses[k].item()
            iteration += 1
        scheduler.step()

        # torchvision detection models only return losses in train mode; disable grads for validation
        val_parts = {k: 0.0 for k in LOSS_KEYS}
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = to_device(images, targets, device)
                for k, v in model(images, targets).items():
                    val_parts[k] += v.item()
        train_parts = {k: v / len(train_loader) for k, v in train_parts.items()}
        val_parts = {k: v / max(len(val_loader), 1) for k, v in val_parts.items()}
        train_loss, val_loss = sum(train_parts.values()), sum(val_parts.values())

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
                with open(ckpt_dir / "best_pq.json", "w") as f:
                    json.dump({"epoch": epoch + 1, "pq": float(pq[b]), "cutoff": cutoffs[b]}, f)
                pq_text += "  (new best)"

        minutes = (time.time() - epoch_start) / 60
        peak = f" peak_gpu_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if device.type == "cuda" else ""
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={minutes:.1f}min{peak}"
              f"{pq_text}", flush=True)
        print(f"    train: {short(train_parts)}\n    val:   {short(val_parts)}", flush=True)
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, round(train_loss, 4), round(val_loss, 4)]
                                   + [round(train_parts[k], 4) for k in LOSS_KEYS]
                                   + [round(val_parts[k], 4) for k in LOSS_KEYS] + [pq_value, pq_cutoff, round(minutes, 2)])

        torch.save(model.state_dict(), ckpt_dir / "last.pt")
        if t.get("save_every") and (epoch + 1) % t["save_every"] == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch + 1}.pt")
        if val_loss < best_val:                                       # kept for comparison; PQ is the better guide
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
