# Run 6 (V6): two single-variable tests targeting small/faint filaments -- neither beat V5

## The idea

Both trained on top of V5's exact recipe (same augmentation, one-annotator-per-epoch, agreement filter, plain
Tversky mask loss unless noted, 30 epochs), motivated by a direct measurement: filaments V5 misses **entirely**
(zero overlap with any prediction) are ~44% smaller in area and ~40% shorter than the ones it catches, while
shape/elongation show no difference at all. Weather/image blur was also checked and ruled out (correlation ~0
with miss rate). Two independent, isolated tests of that finding:

- **6a (size-weighted loss)**: `losses.size_weighted_tversky_mask_loss` -- same BCE + Tversky as V5, but computed
  per-instance and weighted by `(median_batch_area / instance_area) ^ 0.5` (clamped to [0.5, 3.0]) so a small
  filament's mask counts as much toward the loss as a large one's, not as little as its pixel count would
  otherwise give it. Unit-tested (7 tests: gradients flow, weighting is actually applied vs. plain Tversky,
  extreme size ratios stay finite due to clamping, opt-in wiring via `train.mask_loss.type`).
- **6b (anomaly-excluded data)**: identical to V5, but 3 training images excluded
  (`data/splits/train_no_anomalous.txt`) using labels from the external "GONG H-Alpha Anomaly Dataset" (a
  separate, human-labeled good/anomalous image-quality classification -- no filament labels, so no test-answer
  leakage risk, unlike the public MAGFiLO release this project deliberately avoids). 5 of our 707 images are
  flagged anomalous; 2 fall in validation (left untouched) and 3 in train (excluded here). One flagged image
  showed a real, visible defect on inspection (a soft, diffuse limb edge and uneven background).

## Result: neither beats V5

Trained concurrently on one rented H100 ($3.49/hr), sharing the GPU without issue (~50GB/80GB combined, both
completed all 30 epochs with no crashes). Same evaluation protocol (train.py's internal cutoff grid) for a fair,
apples-to-apples comparison across all three:

| Run | Best epoch | PQ (same coarse grid) |
|---|---|---|
| **V5 (baseline)** | 30 | **40.83%** (officially cross-checked, finer grid: 40.92%, see V5/README.md) |
| 6a, size-weighted loss | 20 | 40.60% |
| 6b, anomaly-excluded data | 24 | 39.94% |

6a is close enough to call it noise (-0.23pp) -- this project has repeatedly seen PQ swing 2-3pp between adjacent
evals even within a single run. 6b is more clearly behind (-0.89pp).

**A real process lesson, not just a result:** both runs showed a promising EARLY lead over V5 (6b was +2.6pp
ahead at epoch 6, 6a was +1.2pp ahead at epoch 4) that did NOT hold up by the end of training. Reading early PQ
trends as predictive of the final outcome would have been a mistake here -- worth remembering before declaring
victory early on any future run.

## What this means

Two more single-variable tests join today's other negative/neutral results (EdgeAttNet as a second architecture,
both a naive and an asymmetric two-model ensemble) -- none beat V5. The one real, verified win from today's whole
investigation remains V5 + `merge_radius` post-processing (+0.8pp, see V5/README.md). The pattern across all of
these is consistent: further pipeline/loss/data-cleaning tweaks on the EXISTING training data are hitting
diminishing returns. The dominant remaining failure mode -- 57% of V5's missed filaments have literally nothing
predicted anywhere near them -- is a genuine detection/recall gap that no amount of reweighting or cleaning the
data we already have can fix. Real new training data, specifically more small/faint filament examples, is very
likely necessary for further meaningful progress.

## Files

`checkpoints/run_a_sizeweighted/`, `checkpoints/run_b_no_anomalous/`: `best_pq.pt`, `history.csv`. Not tracked by
git. Checksummed on download (MD5): 6a `17cb056ed9d44a341d972660bd5ab443`, 6b `606d701112da94b5e79c5e3e6e51462b`.

```bash
python scripts/train.py --config V6/configs/maskrcnn_v6a_sizeweighted.yaml
python scripts/train.py --config V6/configs/maskrcnn_v6b_no_anomalous.yaml
```
