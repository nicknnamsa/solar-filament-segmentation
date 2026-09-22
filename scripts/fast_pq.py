"""Fast Panoptic Quality, used for scoring during training and for cutoff sweeps.

Same matching rule as the official metric in metrics.py (a prediction matches a ground-truth filament when IoU > 0.5),
but the IoU matrices are computed once per annotator version with pycocotools and reused for every score cutoff.
Cross-checked against the official code in the model explorer notebook (identical results).
"""
import numpy as np
from pycocotools import mask as mask_util

from metrics import polygon_to_rle


def build_ground_truth(coco, entry_ids):
    """{file_name: {entry_id: [rle, ...]}} for annotator-image entries of a pycocotools COCO object."""
    gt = {}
    for entry_id in entry_ids:
        info = coco.imgs[entry_id]
        rles = [polygon_to_rle(a["segmentation"], info["height"], info["width"]) for a in coco.imgToAnns.get(entry_id, [])]
        gt.setdefault(info["file_name"], {})[entry_id] = rles
    return gt


def sweep_stats(instances, gt, cutoffs, size=(2048, 2048)):
    """Counts pooled over all images and annotator versions, for each score cutoff.

    instances: {file_name: [(score, rle), ...]}, highest score first.
    Returns an array (n_cutoffs, 4): sum of IoU of matches, matched, spurious, missed.
    """
    cutoffs = np.asarray(cutoffs)
    stats = np.zeros((len(cutoffs), 4))
    as_rle = lambda rs: [{"size": list(size), "counts": r.encode("ascii")} for r in rs]
    for file_name, per_entry in gt.items():
        inst = instances[file_name]
        scores = np.array([s for s, _ in inst])
        pred_rles = [r for _, r in inst]
        for gt_rles in per_entry.values():
            n_gt = len(gt_rles)
            if pred_rles and n_gt:
                iou = mask_util.iou(as_rle(pred_rles), as_rle(gt_rles), [0] * n_gt).T      # (n_gt, n_pred)
            else:
                iou = np.zeros((n_gt, len(pred_rles)))
            for c, cut in enumerate(cutoffs):
                k = int((scores >= cut).sum())                                          # predictions kept
                if n_gt == 0:
                    stats[c] += (0, 0, k, 0)
                elif k == 0:
                    stats[c] += (0, 0, 0, n_gt)
                else:
                    hit = iou[:, :k] > 0.5
                    stats[c] += (iou[:, :k][hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum())
    return stats


def pq_from_stats(stats):
    """PQ for each row of `sweep_stats` output."""
    denominator = stats[:, 1] + 0.5 * stats[:, 2] + 0.5 * stats[:, 3]
    return np.divide(stats[:, 0], denominator, out=np.zeros(len(stats)), where=denominator > 0)
