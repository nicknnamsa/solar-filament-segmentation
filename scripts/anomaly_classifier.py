"""A small CNN that classifies a GONG H-alpha image as "good" or "anomalous" (real quality defects -- e.g. a
soft/defocused limb, uneven illumination -- not filaments), trained on the "GONG H-Alpha Anomaly Dataset"
(data/GONG H-Alpha Anomaly Dataset/{train,val,test}/{good,anomalous}/*.jpeg).

This is a genuinely different task from filament segmentation -- image-level binary classification, not
per-pixel instance masks -- so it gets its own small model rather than reusing EdgeAttNet or Mask R-CNN. Images
are resized down (default 256x256): a global "is this image degraded" judgment doesn't need native 2048x2048
detail the way filament boundaries do, and it keeps this trainable in minutes, even on a laptop.

Intended use once trained (not yet wired up): a quality gate over new, unlabeled candidate images before they're
sent for real filament annotation, and/or an auxiliary confidence signal in the prediction pipeline. NOT for
pseudo-labeling filament masks -- this dataset has no filament labels at all.

    python scripts/anomaly_classifier.py --data "data/GONG H-Alpha Anomaly Dataset"
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import get_device


def load_image(path):
    """Grayscale, matching the rest of this project's convention (PIL mode 'L' -> 1 channel)."""
    return torch.from_numpy(np.array(Image.open(path).convert("L"), dtype=np.uint8)).unsqueeze(0)


class AnomalyDataset(Dataset):
    LABELS = {"good": 0.0, "anomalous": 1.0}

    def __init__(self, root, split, resize=256):
        self.samples = []
        for cls, label in self.LABELS.items():
            d = Path(root) / split / cls
            self.samples += [(f, label) for f in sorted(d.iterdir()) if f.suffix.lower() in (".jpeg", ".jpg", ".png")]
        self.resize = resize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = load_image(path).float() / 255.0
        image = F.interpolate(image.unsqueeze(0), size=(self.resize, self.resize), mode="bilinear",
                              align_corners=False).squeeze(0)
        return image, torch.tensor(label, dtype=torch.float32)


class AnomalyClassifier(nn.Module):
    """5 conv/pool blocks (16->32->64->128->128 channels) + global average pool + a single logit. Deliberately
    small -- this is a coarse, whole-image judgment, not per-pixel segmentation."""

    def __init__(self, in_channels=1):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(in_channels, 16), block(16, 32), block(32, 64), block(64, 128),
                                      block(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(1)   # logits, (B,)


def evaluate(model, loader, device):
    model.eval()
    tp = fp = tn = fn = 0
    loss_total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += F.binary_cross_entropy_with_logits(logits, labels, reduction="sum").item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    model.train()
    return {"loss": loss_total / max(n, 1), "accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/GONG H-Alpha Anomaly Dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--checkpoint-dir", default="anomaly_classifier/checkpoints/run1")
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")

    train_ds = AnomalyDataset(args.data, "train", args.resize)
    val_ds = AnomalyDataset(args.data, "val", args.resize)
    test_ds = AnomalyDataset(args.data, "test", args.resize)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)

    model = AnomalyClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall",
                                "val_f1"])

    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.3f} val_precision={val_metrics['precision']:.3f} "
              f"val_recall={val_metrics['recall']:.3f} val_f1={val_metrics['f1']:.3f}", flush=True)
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, round(train_loss, 4), round(val_metrics["loss"], 4),
                                    round(val_metrics["accuracy"], 4), round(val_metrics["precision"], 4),
                                    round(val_metrics["recall"], 4), round(val_metrics["f1"], 4)])

        torch.save(model.state_dict(), ckpt_dir / "last.pt")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), ckpt_dir / "best_f1.pt")

    model.load_state_dict(torch.load(ckpt_dir / "best_f1.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nfinal test set (best_f1.pt): accuracy={test_metrics['accuracy']:.3f} "
          f"precision={test_metrics['precision']:.3f} recall={test_metrics['recall']:.3f} "
          f"f1={test_metrics['f1']:.3f}  (tp={test_metrics['tp']} fp={test_metrics['fp']} "
          f"tn={test_metrics['tn']} fn={test_metrics['fn']})")


if __name__ == "__main__":
    main()
