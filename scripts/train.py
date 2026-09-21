"""Fine-tune Mask R-CNN on the filament training split."""
import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FilamentDataset, collate_fn, read_split
from model import build_model, get_device


def to_device(images, targets, device):
    images = [i.to(device) for i in images]
    targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
    return images, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maskrcnn_baseline.yaml")
    parser.add_argument("--limit", type=int, help="use only N train images (and N//2 val) for a quick smoke test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d, t = cfg["data"], cfg["train"]
    device = get_device()
    print(f"device: {device}")

    def loader(split, shuffle, limit=None):
        ids = read_split(split)[:limit]
        ds = FilamentDataset(d["train_images"], d["annotations"], ids)
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=shuffle,
                          num_workers=t["num_workers"], collate_fn=collate_fn)

    train_loader = loader(d["train_split"], shuffle=True, limit=args.limit)
    val_loader = loader(d["val_split"], shuffle=False, limit=args.limit and max(args.limit // 2, 1))

    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])

    ckpt_dir = Path(t["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(t["epochs"]):
        epoch_start = time.time()
        model.train()
        running = 0.0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch + 1}/{t['epochs']}"):
            images, targets = to_device(images, targets, device)
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        # torchvision detection models only return losses in train mode; disable grads for validation
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = to_device(images, targets, device)
                val_loss += sum(model(images, targets).values()).item()

        train_loss = running / len(train_loader)
        val_loss /= max(len(val_loader), 1)
        peak = f" peak_gpu_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if device.type == "cuda" else ""
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"time={(time.time() - epoch_start) / 60:.1f}min{peak}", flush=True)

        torch.save(model.state_dict(), ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
