"""Turn EdgeAttNet's per-pixel probability map into per-instance (score, rle) predictions, so it can plug into
the same evaluate.py / predict.py / submission pipeline as the instance-segmentation models (V1-V5).

EdgeAttNet has no notion of separate filament instances -- it outputs one "is this pixel part of any filament"
probability map. We threshold it, split it into connected components (each component = one predicted filament),
and score each component by its mean probability. This is a real design decision the paper doesn't make for us
(it's semantic segmentation; no notion of instance confidence exists there), so it's worth being explicit about,
and worth comparing against the alternative (max probability) empirically.

One real consequence worth knowing before reading results: two real, separate filaments that touch or are very
close in the binary mask get merged into ONE connected component here -- a different failure mode from Mask
R-CNN's tendency to fragment one filament into several detections, not obviously better or worse without testing.

`merge_radius` (opt-in, default 0) addresses the OTHER direction of that same coin: one real filament whose
probability map dips below threshold partway along its length gets cut into several disconnected blobs. We
dilate the binary mask by `merge_radius` pixels purely to decide which blobs count as "the same filament" --
the actual returned mask is intersected back with the UNDILATED binary mask, so the submitted mask shape never
grows, only blob identities get merged. This is a real dial with a tradeoff: too large a radius starts merging
genuinely separate, nearby filaments (the failure mode described above) -- tune it empirically against PQ, don't
guess.
"""
import numpy as np
import torch
from pycocotools import mask as mask_util
from scipy import ndimage as ndi


def mask_to_rle(mask):
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


@torch.no_grad()
def predict_instances_edgeattnet(model, image, device, prob_threshold=0.5, min_area=20, score_by="mean",
                                  merge_radius=0):
    """image: (1, H, W) float tensor in [0, 1]. Returns [(score, rle), ...] sorted by descending score, same
    shape as predict.predict_instances, so it's a drop-in for evaluate.py/predict.py.

    merge_radius: pixels to dilate by (4-connected, so `iterations=merge_radius`) before labelling, purely to
    decide which blobs merge -- see module docstring. 0 (default) reproduces the original plain-connected-
    components behaviour exactly.
    """
    logits = model(image.unsqueeze(0).to(device))[0, 0]      # (H, W)
    probs = torch.sigmoid(logits).cpu().numpy()
    binary = probs > prob_threshold

    if merge_radius > 0:
        grouping = ndi.binary_dilation(binary, iterations=merge_radius)
        grouping_labels, n = ndi.label(grouping, structure=np.ones((3, 3)))
        labels = grouping_labels * binary   # zero out dilated-only pixels -- shape stays exactly `binary`
    else:
        labels, n = ndi.label(binary, structure=np.ones((3, 3)))  # 8-connectivity: touching diagonally counts

    instances = []
    for label_id in range(1, n + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < min_area:
            continue
        score = float(probs[component].mean() if score_by == "mean" else probs[component].max())
        instances.append((score, mask_to_rle(component)))
    return sorted(instances, key=lambda x: -x[0])
