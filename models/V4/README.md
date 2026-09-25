# Run 4 (V4): everything with proven positive impact, bundled

## The idea

Not a new hypothesis — a consolidation. Across V1-V3 plus the free experiments (checkpoint ensembling, TTA,
denoising, a barb-focused loss diagnostic), only two things showed a real positive effect:

1. **V2's recipe over V1** (+1 to 1.6 points): augmentation, one-annotator-per-epoch sampling, the agreement
   filter (drop filaments no other annotator drew), PQ-based checkpoint selection instead of validation loss.
2. **TTA** (flip/rotation-averaged predictions), +0.2 points — noise-level, and a prediction-time technique, not
   a training change, so it's applied when generating predictions, not baked into this run.

Everything else tested came back negative and is deliberately excluded: limb correction (V1), the widened
box/RPN IoU margins (V3, a regression), the checkpoint ensemble of V1+V1+V2 (worse than V2 alone), denoising
(no separable noise found), and a spine-weighted barb loss (dropped before building it — our own diagnostic
showed barb-heavy filaments are missed at about the same rate as simple ones, so there was no basis for it).

V4 is V2's recipe, cleanly re-run, with two small refinements: 30 epochs instead of 40 (V2 peaked at epoch
18-20; epochs 30-40 only kept overfitting), and originally `eval_every: 1` for finer-grained checkpoint
selection — reverted to `eval_every: 2` mid-run when this run's pod turned out to have an unusually slow
per-epoch evaluation step (see below), to keep cost under control.

## A infrastructure problem, not a modelling one

This run's pod initially trained at 5-6 minutes/epoch with the GPU sitting near 0% utilization — `/workspace`
on that host was mounted over the network, and every epoch was re-reading all 601 training images from it.
Copying the data to the container's local disk (`/root/solar_local`, `/dev/vda1`) instead of the network-backed
volume dropped epoch time to under a minute and GPU utilization to 88-96%, a roughly 6x speedup. Not a training
result, just a note for future runs: verify `/workspace` isn't network-mounted early (`df -h`), or default to
copying data locally before training regardless.

## Result

Trained on a rented H100 (45 minutes for 30 epochs, once running on local disk). Same 106 held-out validation
images, same protocol as V1-V3 (official metric, best score cutoff, cross-checked; `model_explorer.ipynb`
section 6):

| Model | Best cutoff | PQ | Bootstrap 5-95% range | Honest estimate |
|---|---|---|---|---|
| V1 raw, epoch 20 | 0.800 | 38.6% | 36.5 to 40.4 | 38.4 +/- 1.3 |
| V3 final, epoch 40 | 0.875 | 38.6% | 36.5 to 40.5 | 38.3 +/- 1.3 |
| V2 final, epoch 40 | 0.875 | 39.6% | 37.6 to 41.5 | 39.4 +/- 1.3 |
| V3 best by PQ, epoch 20 | 0.900 | 39.4% | 37.1 to 41.4 | 39.0 +/- 1.3 |
| **V4 final, epoch 30** | 0.825 | 39.4% | 37.1 to 41.6 | 39.1 +/- 1.5 |
| V2 best by PQ, epoch 18 | 0.850 | 40.2% | 38.0 to 42.2 | 39.9 +/- 1.4 |
| **V4 best by PQ, epoch 20** | 0.875 | **40.5%** | 38.3 to 42.5 | 40.3 +/- 1.4 |

**V4 is the best checkpoint across all four runs, but the margin over V2 (+0.3) is well within this
106-image validation set's noise (ranges of +/-1.3 to 1.9).** Read this as "confirms V2's recipe is solid and
reproducible," not as a proven further improvement — the two runs are statistically indistinguishable.

## Files

`checkpoints/run1/`: `best_pq.pt` (epoch 20), `last.pt`, `epoch_10/20/30.pt`, `history.csv`, `best_pq.json`,
`config.yaml`. `visualizations/cutoff_curve_*.png`. Not tracked by git.

```bash
python scripts/train.py --config V4/configs/maskrcnn_v4.yaml
python scripts/evaluate.py --config V4/configs/maskrcnn_v4.yaml --checkpoint V4/checkpoints/run1/best_pq.pt
```
