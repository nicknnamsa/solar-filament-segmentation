"""Optional recall-biased mask loss for Mask R-CNN, via monkey-patching torchvision's internal loss function.

Why: our error analysis (see V4/README.md) split "missed" filaments into two groups the mask loss can plausibly
fix: ~18% are detected with a badly incomplete mask (IoU 0.01-0.3), and ~41% are detected with a mask that is
close but falls just short of the 0.5 IoU matching threshold (IoU 0.3-0.5). Standard BCE treats every pixel
equally; a Tversky loss lets us penalize under-covering the true mask (false negatives) more than over-covering
it (false positives), biasing the network toward fuller, more generous masks -- aimed at nudging those
close-but-not-quite cases over the threshold. (A separate ~38% of misses are correctly-shaped detections that
scored under our cutoff -- a confidence problem, not a mask-shape one; that's what TTA targets instead, see
tta.py and V4/README.md.)

torchvision's Mask R-CNN calls a plain module-level function, `maskrcnn_loss`, to compute the mask loss; it is
looked up by name at call time, so replacing that name in its module swaps the loss for every model built
afterwards in this process. This is opt-in: call `patch_mask_loss(cfg)` once per run; with no `mask_loss` key in
the config it is a no-op, so every existing config (V1-V4) trains exactly as before.
"""
import torch
import torch.nn.functional as F
from torchvision.models.detection import roi_heads
from torchvision.models.detection.roi_heads import project_masks_on_boxes

_ORIGINAL_MASKRCNN_LOSS = roi_heads.maskrcnn_loss   # kept so tests / other code can restore it


def tversky_mask_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, alpha=0.3, beta=0.7,
                       bce_weight=0.5, tversky_weight=0.5, smooth=1.0):
    """Drop-in replacement for torchvision's maskrcnn_loss: BCE + Tversky, both on the same per-pixel logits.

    alpha weights false positives (predicted-but-not-real), beta weights false negatives (real-but-missed).
    beta > alpha biases toward more complete ("generous") masks -- the recall-favoring direction we want, since
    our own error analysis shows more of our score comes from missed area than from excess predicted area.
    alpha + beta = 1 is the usual convention (Salehi et al. 2017, the original Tversky loss paper, uses 0.3/0.7).
    """
    discretization_size = mask_logits.shape[-1]
    labels = torch.cat([gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)], dim=0)
    mask_targets = torch.cat(
        [project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)],
        dim=0,
    )
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
    bce = F.binary_cross_entropy_with_logits(logits, mask_targets)

    probs = torch.sigmoid(logits)
    tp = (probs * mask_targets).sum()
    fp = (probs * (1 - mask_targets)).sum()
    fn = ((1 - probs) * mask_targets).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    return bce_weight * bce + tversky_weight * (1 - tversky)


def size_weighted_tversky_mask_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, alpha=0.3,
                                     beta=0.7, bce_weight=0.5, tversky_weight=0.5, smooth=1.0, weight_power=0.5,
                                     min_weight=0.5, max_weight=3.0):
    """Same idea as tversky_mask_loss (BCE + Tversky), but additionally upweights SMALLER ground-truth instances.

    Motivated by a direct measurement, not a guess (see V5/README.md's data-diagnosis addendum): filaments V5
    misses entirely are ~44% smaller in area and ~40% shorter than the ones it catches, while shape/elongation
    show no difference at all. Plain BCE/Tversky (above) reduce over ALL instances' pixels in a batch together,
    so a few large filaments' pixels dominate the gradient and a small filament barely registers. This computes
    the loss PER INSTANCE first, then weights each instance's contribution by (median_batch_area / its_area) ^
    weight_power (clamped to [min_weight, max_weight] so one tiny/noisy mask can't dominate the batch), so a
    small filament counts as much as a large one, not as little as its pixel count would otherwise give it.

    Note: area here is measured in the discretized mask-head's own resolution (e.g. 28x28), not original
    2048x2048 image pixels like the diagnostic used -- relative ordering (smaller vs larger) is preserved by the
    resize, which is all this needs, but the absolute numbers aren't directly comparable across the two.
    """
    discretization_size = mask_logits.shape[-1]
    labels = torch.cat([gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)], dim=0)
    mask_targets = torch.cat(
        [project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)],
        dim=0,
    )
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
    probs = torch.sigmoid(logits)

    areas = mask_targets.flatten(1).sum(1).clamp(min=1.0)   # (N,) pixel area per instance, discretized mask space
    median_area = areas.median()
    weights = (median_area / areas).clamp(min_weight, max_weight) ** weight_power
    weights = weights / weights.mean()   # keep the overall loss scale comparable to the unweighted version

    bce_per_instance = F.binary_cross_entropy_with_logits(logits, mask_targets, reduction="none").flatten(1).mean(1)
    bce = (weights * bce_per_instance).mean()

    tp = (probs * mask_targets).flatten(1).sum(1)
    fp = (probs * (1 - mask_targets)).flatten(1).sum(1)
    fn = ((1 - probs) * mask_targets).flatten(1).sum(1)
    tversky_per_instance = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    tversky_term = (weights * (1 - tversky_per_instance)).mean()

    return bce_weight * bce + tversky_weight * tversky_term


_LOSS_FNS = {"tversky": tversky_mask_loss, "size_weighted_tversky": size_weighted_tversky_mask_loss}
_LOSS_KWARGS = {
    "tversky": ("alpha", "beta", "bce_weight", "tversky_weight"),
    "size_weighted_tversky": ("alpha", "beta", "bce_weight", "tversky_weight", "weight_power", "min_weight",
                              "max_weight"),
}


def patch_mask_loss(cfg):
    """Reads train.mask_loss from the config; monkey-patches torchvision's mask loss if present, else does nothing.

    Config (all optional, shown with the recall-biased defaults used if the section exists but leaves them out):
      train:
        mask_loss:
          type: tversky                # or size_weighted_tversky (see size_weighted_tversky_mask_loss)
          alpha: 0.3                    # false-positive weight
          beta: 0.7                     # false-negative weight (higher = more generous masks)
          bce_weight: 0.5
          tversky_weight: 0.5
          weight_power: 0.5             # size_weighted_tversky only
          min_weight: 0.5               # size_weighted_tversky only
          max_weight: 3.0               # size_weighted_tversky only
    """
    spec = cfg.get("train", {}).get("mask_loss")
    if not spec:
        return False
    loss_type = spec.get("type", "tversky")
    if loss_type not in _LOSS_FNS:
        raise ValueError(f"unknown mask_loss type: {loss_type!r} (expected one of {list(_LOSS_FNS)})")
    kwargs = {k: spec[k] for k in _LOSS_KWARGS[loss_type] if k in spec}
    base_fn = _LOSS_FNS[loss_type]

    def loss_fn(*args, **inner_kwargs):
        return base_fn(*args, **inner_kwargs, **kwargs)

    roi_heads.maskrcnn_loss = loss_fn
    print(f"mask loss: {loss_type} (patched), {kwargs or 'defaults'}")
    return True
