"""Test-time augmentation: pool predictions from the 8 dihedral views (flips/90-degree rotations) of an image.

Shared by scripts/tta_ensemble.py (which measures whether this helps, on the validation set) and predict.py
(--tta flag, which uses it to generate actual submissions). See V4/README.md's TTA section for the measured effect.
"""
import numpy as np
import torch
from pycocotools import mask as mask_util

from predict import predict_instances

SIZE = (2048, 2048)


def dihedral_forward_image(image, i):
    """i in 0..7: the 8 symmetries of a square. image is a (C, H, W) tensor."""
    k, flip = i % 4, i >= 4
    x = image.flip(-1) if flip else image
    return torch.rot90(x, k, (1, 2))


def dihedral_inverse_mask(mask, i):
    """Undo dihedral_forward_image on a (H, W) numpy mask."""
    k, flip = i % 4, i >= 4
    x = np.rot90(mask, -k)
    return np.fliplr(x) if flip else x


def cluster_instances(items, iou_thresh):
    """items: [(score, rle, source_id), ...]. Returns [(rle, max_score, n_distinct_sources), ...], one per cluster,
    using each cluster's highest-scoring member's mask."""
    n = len(items)
    if n == 0:
        return []
    rles = [{"size": list(SIZE), "counts": r.encode("ascii")} for _, r, _ in items]
    iou = mask_util.iou(rles, rles, [0] * n)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if iou[i, j] >= iou_thresh:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        best = max(idx, key=lambda i: items[i][0])
        out.append((items[best][1], items[best][0], len({items[i][2] for i in idx})))
    return out


def predict_instances_tta(model, image, device, mask_threshold=0.5, cluster_iou=0.3, min_score=0.05):
    """Drop-in replacement for predict.predict_instances that pools 8 dihedral views.

    Returns [(score, rle), ...] sorted by descending score, same shape as the non-TTA function, so it can be
    used anywhere predict_instances is used (predict.py, evaluate.py).
    """
    pooled = []
    for i in range(8):
        transformed = dihedral_forward_image(image, i)
        for score, rle in predict_instances(model, transformed, device, min_score=min_score, mask_threshold=mask_threshold):
            mask = mask_util.decode({"size": list(SIZE), "counts": rle.encode("ascii")})
            restored = np.asfortranarray(dihedral_inverse_mask(mask, i))
            pooled.append((score, mask_util.encode(restored)["counts"].decode("ascii"), i))
    clusters = cluster_instances(pooled, cluster_iou)
    clusters.sort(key=lambda c: -c[1])   # (rle, score, support), highest score first

    # Clustering only merges DUPLICATES of the same filament across views (IoU >= cluster_iou); it says nothing
    # about two DIFFERENT filaments' final masks sharing pixels, which Kaggle's submission format forbids (and
    # our own PQ scorer doesn't check, since the metric itself doesn't penalize it -- this went unnoticed until
    # a real submission was rejected). Enforce it the same way predict.predict_instances does: greedy, highest
    # score keeps contested pixels, and a cluster that becomes empty is dropped.
    taken = np.zeros(SIZE, dtype=bool)
    out = []
    for rle, score, _ in clusters:
        mask = mask_util.decode({"size": list(SIZE), "counts": rle.encode("ascii")}).astype(bool) & ~taken
        if mask.any():
            taken |= mask
            out.append((score, mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")))
    return out
