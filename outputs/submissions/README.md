# Submissions

Files to upload to Kaggle, and a record of how each was made.

## What to upload

**`v2_run1_best_pq_cutoff0.85.csv`**: upload this on the competition's *Submit Predictions* page
(https://www.kaggle.com/competitions/filament-segmentation-2026).

- Columns: `filament_id`, `segmentation_rle`. One row per predicted filament.
- `filament_id` is `<image name>_<number>`, for example `20110120105534Ch_0`.
- `segmentation_rle` is a compressed COCO run-length string for a 2048x2048 mask (the format the organisers' scoring
  notebook decodes). Filaments in one image do not overlap.
- 1,126 filaments over 177 of the 180 test images. Three images have no rows (nothing scored above the cutoff).
  If Kaggle requires a row for every image, check its `sample_submission.csv` for how empty images are written.

## Log

| File | Model | Cutoff | Validation PQ | Leaderboard |
|---|---|---|---|---|
| `v2_run1_best_pq_cutoff0.85.csv` | V2 run 1, epoch 18 (`V2/checkpoints/run1/best_pq.pt`) | 0.85 | 40.2% (range 38.0 to 42.2; honest estimate 39.9 +/- 1.4) | 0.36 |
| `v4_run1_best_pq_cutoff0.875.csv` | V4 run 1, epoch 20 (`V4/checkpoints/run1/best_pq.pt`) | 0.875 | 40.5% (range 38.3 to 42.5; honest estimate 40.3 +/- 1.4) | 0.37 |
| `v4_run1_best_pq_tta_cutoff0.9.csv` | V4 run 1, epoch 20, +TTA (8-view pooling, `scripts/tta.py`) | 0.9 | 40.8% | 0.36 |
| `v5_run1_best_pq_cutoff0.825.csv` | V5 run 1, epoch 30, Tversky mask loss (`V5/checkpoints/run1/best_pq.pt`) | 0.825 | 40.92% (range 38.8 to 42.8; honest estimate 40.7 +/- 1.4), see `V5/README.md` | 0.36 |
| `v5_run1_merge2_cutoff0.825.csv` | V5 run 1, epoch 30, + merge-radius=2 post-processing (`scripts/predict.py`) | 0.825 | **41.73%**, cross-checked exactly against the official scorer -- current best, see `V5/README.md`'s merge-radius section | *(fill in)* |

Validation PQ is on the 106 held-out validation images with the official metric. Expect the leaderboard number to
differ: the hidden test set may be scored differently (for example how several annotators are treated).

## How this file was made

```bash
python scripts/predict.py --config V2/configs/maskrcnn_v2.yaml \
    --checkpoint V2/checkpoints/run1/best_pq.pt --split test --score-threshold 0.85 \
    --out submissions/v2_run1_best_pq_cutoff0.85.csv

python scripts/predict.py --config V4/configs/maskrcnn_v4.yaml \
    --checkpoint V4/checkpoints/run1/best_pq.pt --split test --score-threshold 0.875 \
    --out submissions/v4_run1_best_pq_cutoff0.875.csv
```

Raw 2048x2048 images, mask threshold 0.5, overlaps between filaments removed (higher score wins). The same code
path on the validation images, scored with the official code, gave 40.17% for V2 and 40.54% for V4 (matches the
numbers above). Predictions may differ slightly if re-run on different hardware.

Checksums (MD5): v2 CSV `3ac5ba7206af6c265f1b162d3c70bf35`, v2 checkpoint `c9723d82092e5dae13ee2d375379333c`.

**Note on the TTA submission:** the first version generated for V4 was rejected by Kaggle for overlapping masks
-- a real bug in `scripts/tta.py`, fixed and re-verified (see `V5/README.md`). `v4_run1_best_pq_tta_cutoff0.9.csv`
above is the corrected file; every submission listed here has been checked for zero mask overlap across every
image before being handed off.

**`v5_run1_merge2_cutoff0.825.csv`**: 1062 filaments over 175 of 180 test images, generated with
`python scripts/predict.py --config V5/configs/maskrcnn_v5.yaml --checkpoint V5/checkpoints/run1/best_pq.pt
--split test --score-threshold 0.825 --merge-radius 2 --out submissions/v5_run1_merge2_cutoff0.825.csv`.
Checked for zero mask overlap across all 175 non-empty images before being handed off. See `V5/README.md` for
where the merge-radius idea came from and its official-scorer verification.

## Still needed

The competition rules require an end-to-end notebook (see the repo plan). `notebooks/pipeline_walkthrough.ipynb` is
still an empty skeleton, so it is not ready to submit.
