# Run 3 (V3): padded "ignore margin" instead of forcing near-miss regions to be background

## The idea

Mask R-CNN's box-classification stage currently has no slack at all: any candidate region overlapping a real
filament by less than 50% IoU is forced to count as confirmed background during training. Annotators only agree
with each other about 34% of the time (`scripts/annotator_agreement.py`), so a lot of "background" near a real
filament is genuinely ambiguous, not empty. This run widens the existing ignore band torchvision's matcher already
supports (a region scoring between the two thresholds is dropped from the loss instead of being forced either way),
so a near-miss is no longer punished as a confident mistake. Single-variable change from V2: everything else
(images, augmentation, one-annotator-per-epoch sampling, the agreement filter, checkpoint selection by PQ) is
identical. See `scripts/model.py` for the change and its verification.

| | V2 | V3 |
|---|---|---|
| `box_bg_iou_thresh` | 0.5 (== fg threshold, no margin) | 0.3 |
| `rpn_bg_iou_thresh` | 0.3 | 0.2 |

## Result

Trained on a rented H100 (53 minutes for 40 epochs). Same 106 held-out validation images, same protocol as V1/V2
(official metric, best score cutoff, cross-checked; `model_explorer.ipynb` section 6):

| Model | Best cutoff | PQ | Bootstrap 5-95% range | Honest estimate |
|---|---|---|---|---|
| V2 best by PQ, epoch 18 | 0.850 | 40.2% | 38.0 to 42.2 | 39.9 +/- 1.4 |
| V2 final, epoch 40 | 0.875 | 39.6% | 37.6 to 41.5 | 39.4 +/- 1.3 |
| **V3 best by PQ, epoch 20** | 0.900 | **39.4%** | 37.1 to 41.4 | 39.0 +/- 1.3 |
| **V3 final, epoch 40** | 0.875 | **38.6%** | 36.5 to 40.5 | 38.3 +/- 1.3 |

**This did not help.** V3 scores about 0.8-1.0 points below V2 at both comparison points, within the noise of this
106-image validation set but consistently in the wrong direction, not the right one.

## Why, best guess

Comparing detection counts at each run's best cutoff: V2 (696 matched, 360 spurious, 562 missed) vs V3 (648 matched,
279 spurious, 610 missed). V3 makes fewer confident false alarms, but also fewer true positives and more misses —
it became more conservative, not more accurate. Mask overlap of matches (SQ) was unchanged (0.66-0.67 in both).
Removing the "near-miss = confirmed wrong" training pressure seems to have made the classifier less decisive near
real filaments generally, rather than selectively rescuing the plausible-but-unlabelled cases we were hoping for.

## Files

`checkpoints/run1/`: `best_pq.pt` (epoch 20), `last.pt`, `epoch_10/20/30/40.pt`, `history.csv`, `best_pq.json`,
`config.yaml`. `visualizations/cutoff_curve_*.png`. Not tracked by git.

```bash
python scripts/train.py --config V3/configs/maskrcnn_v3.yaml
python scripts/evaluate.py --config V3/configs/maskrcnn_v3.yaml --checkpoint V3/checkpoints/run1/best_pq.pt
```
