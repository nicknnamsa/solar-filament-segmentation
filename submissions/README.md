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
| `v2_run1_best_pq_cutoff0.85.csv` | V2 run 1, epoch 18 (`V2/checkpoints/run1/best_pq.pt`) | 0.85 | 40.2% (range 38.0 to 42.2; honest estimate 39.9 +/- 1.4) | *(fill in)* |

Validation PQ is on the 106 held-out validation images with the official metric. Expect the leaderboard number to
differ: the hidden test set may be scored differently (for example how several annotators are treated).

## How this file was made

```bash
python scripts/predict.py --config V2/configs/maskrcnn_v2.yaml \
    --checkpoint V2/checkpoints/run1/best_pq.pt --split test --score-threshold 0.85 \
    --out submissions/v2_run1_best_pq_cutoff0.85.csv
```

Raw 2048x2048 images, score cutoff 0.85, mask threshold 0.5, overlaps between filaments removed (higher score wins).
The same code path on the validation images, scored with the official code, gave 40.17% (matches the number above).
Predictions may differ slightly if re-run on different hardware.

Checksums (MD5): CSV `3ac5ba7206af6c265f1b162d3c70bf35`, checkpoint `c9723d82092e5dae13ee2d375379333c`.

## Still needed

The competition rules require an end-to-end notebook (see the repo plan). `notebooks/pipeline_walkthrough.ipynb` is
still an empty skeleton, so it is not ready to submit.
