# Run 2 (V2)

Changes from run 1 (`V2/configs/maskrcnn_v2.yaml`), all driven by what run 1 showed:

| Change | Why |
|---|---|
| Score PQ on the validation images every 2 epochs; keep the best epoch by PQ (`best_pq.pt`, `best_pq.json`) | Validation loss picked the wrong checkpoint in run 1 |
| Loss written out by component, plus `history.csv` per run | See which loss terms rise when validation loss climbs |
| Score cutoff 0.8 | Best cutoff in run 1 (flat between about 0.55 and 0.875) |
| Raw images only | Limb correction gave no gain |
| Flips, 90-degree rotations, brightness/contrast jitter | Run 1 overfit: training loss 0.39, validation loss 1.1 |
| One randomly chosen annotator's version of each image per epoch | Run 1 saw the same image up to 3 times with conflicting masks |
| Drop filaments no other annotator drew (IoU >= 0.3), for images with 2+ annotators | Such a filament is a false alarm against the other annotators under the scoring; drops 12.8% of training filaments |
| Mixed precision, batch size 4 (lr 0.01), 200 warmup iterations | Faster runs; one run 1 job used only 8 GB of 24 GB |

Not changed: model architecture, input size, the train/validation split (so results are comparable with V1).

## Result

Trained on a rented H100 (about 46 minutes for 40 epochs, roughly $4.50 including setup). Panoptic Quality on the
same 106 held-out validation images as run 1, at the best score cutoff, scored with the official metric
(`model_explorer.ipynb`, section 6; the fast scorer was cross-checked against the official code each time):

| Model | Best cutoff | PQ | Bootstrap 5-95% range | Honest estimate (cutoff tuned on half, scored on the other half) |
|---|---|---|---|---|
| V1 raw, epoch 20 | 0.800 | 38.6% | 36.5% to 40.4% | 38.4% +/- 1.3 |
| V2 final, epoch 40 (`last.pt`) | 0.875 | 39.6% | 37.6% to 41.5% | 39.4% +/- 1.3 |
| V2 best by PQ, epoch 18 (`best_pq.pt`) | 0.850 | 40.2% | 38.0% to 42.2% | 39.9% +/- 1.4 |

- The gain over run 1 is about +1.0 point (final epoch) to +1.6 points (epoch 18). The ranges overlap, so this is a
  small improvement, not a proven one. Epoch 18 was chosen using the same validation images it is scored on, so its
  number is a little optimistic; the final-epoch row is the cleaner comparison.
- Overfitting is much lower in loss terms: validation loss stayed near 0.85 (run 1 climbed to 1.1) at a final training
  loss of 0.59.
- Detection improved slightly on all three counts at the best cutoff (matched 696, spurious 360, missed 562 for the epoch-18
  model, against 680 / 378 / 578 for run 1), and mask overlap of matches rose from 0.66 to 0.67.
- The best cutoff moved up (0.85 to 0.875 vs 0.8): this model's confidence scores are sharper. PQ is within 1 point of the
  best anywhere from about 0.775 to 0.875. At a fixed cutoff of 0.5 it is worse than run 1, so use 0.85, not 0.5.
- `history.csv` has the per-epoch loss parts and PQ; `visualizations/cutoff_curve_*.png` show PQ against cutoff.

## Files

`checkpoints/run1/`: `best_pq.pt`, `last.pt`, `epoch_10/20/30/40.pt`, `best.pt` (lowest validation loss, not recommended),
`history.csv`, `best_pq.json`, `config.yaml` (the config as run; `num_workers` was 20 on the pod). Logs in `logs/`.
Checkpoints are not tracked by git.

```bash
python scripts/train.py --config V2/configs/maskrcnn_v2.yaml
python scripts/evaluate.py --config V2/configs/maskrcnn_v2.yaml --checkpoint V2/checkpoints/run1/best_pq.pt
```

Mixed precision and lr 0.01 worked on the GPU (no divergence). Two things learned about running on a fast GPU: the
loader, not the GPU, was the bottleneck (fixed with `fast_loader`), and scoring PQ inside training must skip
low-score detections or it costs minutes per check.
