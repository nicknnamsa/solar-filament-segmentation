# Run 1 (V1): baseline Mask R-CNN, raw vs limb-corrected images

The first training round. Two models, identical apart from the input images, trained for 20 epochs each on a rented
RTX 4090 (both at once on the same GPU, about 2 hours, roughly $3.30 including setup and scoring).

| | Raw images | Limb-darkening-corrected images |
|---|---|---|
| Config | `configs/maskrcnn_baseline.yaml` | `configs/maskrcnn_limb.yaml` |
| Checkpoints | `checkpoints/raw/` | `checkpoints/limb/` |
| Training log | `logs/raw.log` | `logs/limb.log` |

Both used: Mask R-CNN (ResNet-50 FPN, COCO-pretrained), 2048x2048 input, batch size 2, SGD lr 0.005 with cosine decay,
every annotator's version of an image as a separate training sample, no augmentation.

## Results

Panoptic Quality on the 106 held-out validation images (official metric, best of 7 score cutoffs; see `logs/eval_*.log`):

| Checkpoint | Raw | Limb-corrected |
|---|---|---|
| `best.pt` (lowest validation loss: epoch 5 raw, epoch 2 limb) | 35.5% | 33.3% |
| `last.pt` (epoch 20) | **38.6%** | **38.6%** |

Best score cutoff was 0.8 for both final models. A finer sweep of the raw final model (notebook `model_explorer.ipynb`,
section 6) gives 38.6% at 0.8 with a bootstrap range of 36.5% to 40.4%, PQ within 1 point of the best anywhere from
0.55 to 0.875, and an honest half-and-half estimate of 38.4% +/- 1.3.

## What we learned

- Limb-darkening correction made no measurable difference (38.6% vs 38.6%).
- Validation loss rose from early on (training loss 0.39 vs validation loss 1.1 at epoch 20) while PQ still improved, so
  the "lowest validation loss" checkpoint is the wrong one to keep. Choose epochs by PQ.
- Annotators agree with each other at only about 34% PQ (`scripts/annotator_agreement.py`), so the labels themselves cap
  what is achievable. The model's masks overlap the annotators' about as well as they overlap each other
  (mean IoU of matches about 0.65 vs 0.63); most of the remaining headroom is in detection (missed and spurious
  filaments), not mask shape.
- Long filaments are sometimes predicted in several pieces. Merging nearby pieces by distance alone gained only
  about 0.6 points (within noise), so it was not adopted.

## Reproducing

The code is shared (`scripts/`); the run-1 configs still work unchanged with it.

```bash
python scripts/preprocess.py                                    # limb-corrected images -> data/processed/limb/ (limb run only)
python scripts/train.py --config V1/configs/maskrcnn_baseline.yaml
python scripts/evaluate.py --config V1/configs/maskrcnn_baseline.yaml --checkpoint V1/checkpoints/raw/last.pt
```

`checkpoint_dir` in these configs was updated to point into `V1/` when the repo was reorganised. `score_threshold` is
still the original 0.5; use 0.8 for predictions.

Checkpoints (`*.pt`, 176 MB each) are not tracked by git.
