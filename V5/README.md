# Run 5 (V5): recall-biased Tversky mask loss

## The idea

Single-variable test on top of V4's recipe (our best so far): a Tversky loss added to the mask branch (on top of
standard BCE), weighted to penalize under-covering a filament's true mask (false negatives, weight 0.7) more than
over-covering it (false positives, weight 0.3). See `scripts/losses.py`, which monkey-patches torchvision's
internal mask loss function -- opt-in only, so V1-V4 train exactly as before.

Motivated by a direct measurement on V4's misses (not a guess): of the ~575 filaments V4 missed on the
validation set, only 2% had literally nothing detected there. The rest split into two real, different problems:
- **~59%** were detected with an imprecise mask -- 18% badly wrong (IoU 0.01-0.3), 41% *close* but just short of
  the 0.5 IoU matching threshold (IoU 0.3-0.5), both scored fairly confidently (~0.77-0.80). This is what the
  Tversky loss targets: nudge these over the line by making the network draw fuller, more generous masks.
- **~38%** were detected with a *correct* mask (IoU > 0.5!) that simply scored under our cutoff (mean score
  0.61) -- a confidence problem, not a mask-shape problem. That's what TTA targets instead (see below).

An earlier idea in this direction (a spine-weighted loss, using the barb annotations directly) was dropped before
being built: a diagnostic showed barb-heavy filaments aren't missed more than simple ones, so there was no basis
for it. This Tversky loss targets the *general* imprecise-mask problem instead, not barbs specifically.

## Result

Trained on a rented H100 (34 minutes for 30 epochs, once running on local disk -- this pod's `/workspace` was
also network-mounted, like V4's; data was uploaded straight to `/dev/vda1` local storage from the start this
time instead of discovering the problem mid-run). Unlike every previous run, PQ kept climbing right through
epoch 30 with no sign of the overfitting pattern V1-V4 showed after epoch ~20 -- `best_pq.pt` and `last.pt` are
the same checkpoint here.

Same 106 held-out validation images, same protocol (official metric, cross-checked exactly):

| Model | Best cutoff | PQ | Bootstrap 5-95% range | Honest estimate |
|---|---|---|---|---|
| V4 best by PQ | 0.875 | 40.5% | 38.3 to 42.5 | 40.3 +/- 1.4 |
| V4 + TTA | 0.9 | 40.8% | -- | -- |
| **V5, epoch 30** | **0.825** | **40.92%** | 38.8 to 42.8 | 40.7 +/- 1.4 |
| V5 + TTA | 0.9 | 40.9% | -- | -- |

**V5 is our best model.** The gain over V4 (+0.4) is still within this validation set's noise, but it's the
most diagnostically-justified single-variable improvement since V2, and it shows up consistently: higher
in-training PQ than V4 at every comparable checkpoint, not just at the final one.

**TTA adds nothing on top of V5** (40.92% -> 40.9%, statistically identical), unlike on V4 where it added a real
+0.3. Reading this together with the diagnosis above: TTA was rescuing the confidence-calibration failures (the
38% group). V5's fuller masks apparently also come with better-calibrated confidence naturally, leaving little
for TTA to additionally rescue. The two fixes overlap rather than stack. **Practical conclusion: use V5 alone --
it's simpler (no 8x TTA inference cost) and no worse.**

## Post-processing addendum: merging nearby detections (+0.8pp, free)

Found while diagnosing V5 against a separate architecture (EdgeAttNet, see `EdgeAttNet/README.md` -- that
experiment underperformed V5 and was retired, but the diagnostic method it forced us to build paid off here).
Decomposing PQ = SQ x RQ for V5's predictions (cutoff 0.825): SQ 66.6%, RQ 61.5%, matched=716, FP=356, FN=542.
Checking whether V5's own false positives/negatives are genuine misses or near-misses of a real filament: **78%**
of V5's 356 spurious predictions still have *some* nonzero overlap with a real filament -- i.e. Mask R-CNN
sometimes emits multiple adjacent/overlapping-but-not-quite-matching detections for what should be one filament,
the same fragmentation pattern (just smaller) that EdgeAttNet's connected-components step showed much more
severely. (The other side of that coin: 57% of V5's 542 missed filaments have *zero* overlap with anything
predicted -- a genuine detection/recall gap that no post-processing can fix; see `EdgeAttNet/README.md`'s
complementary-error findings and the project's current data-lever discussion.)

`predict.merge_nearby_instances` (applied AFTER score-thresholding): dilate each kept instance's mask by
`radius` pixels, union any whose dilated versions touch/overlap into one instance (using the union of their
*original*, undilated pixels -- the submitted shape never grows, only identities merge), scored by the group's
max score. Measured on the validation set:

| radius | PQ | SQ | RQ | matched | FP | FN |
|---|---|---|---|---|---|---|
| 0 (V5 alone) | 40.9% | 66.6% | 61.5% | 716 | 356 | 542 |
| 1 through 12 | **41.7%** | 66.5% | 62.8% | 707 | **287** | 551 |

Stable across the entire radius range tested (1-12 give identical results) -- not a fragile, over-tuned choice.
Net effect: a few marginal matches are traded away (716->707) for a much larger cut in false positives
(356->287), a clear net win on PQ. Cross-checked against the official scorer at radius=2: fast_pq 41.73% vs.
official 41.73%, exact match.

```bash
python scripts/predict.py --config V5/configs/maskrcnn_v5.yaml --checkpoint V5/checkpoints/run1/best_pq.pt \
    --split test --score-threshold 0.825 --merge-radius 2 --out submissions/v5_run1_merge2_cutoff0.825.csv
```

## A process note: a real bug, caught by a real Kaggle rejection

The first TTA submission (for V4) was rejected by Kaggle: *"Submissions may not contain overlapping masks."*
`predict_instances_tta` (`scripts/tta.py`) clustered duplicate detections of the *same* filament across the 8
views correctly, but never checked that *different* filaments' final masks didn't share pixels -- unlike the
plain (non-TTA) prediction path, which has always enforced this. Our own PQ scorer doesn't catch this either,
since the metric itself doesn't penalize overlapping predictions; only Kaggle's submission format does. Fixed by
adding the same greedy non-overlap step the plain path already uses. Verified with a synthetic test (two
detections with real overlap below the merge threshold; confirmed the higher-scoring one keeps the contested
pixels and the lower one is trimmed) and by decoding and checking every image in both regenerated submissions
(zero overlap, all 175-177 non-empty images each) before handing them off again.

## Files

`checkpoints/run1/`: `best_pq.pt` (== `last.pt`, epoch 30), `epoch_10/20/30.pt`, `history.csv`, `best_pq.json`,
`config.yaml`. Not tracked by git.

```bash
python scripts/train.py --config V5/configs/maskrcnn_v5.yaml
python scripts/predict.py --config V5/configs/maskrcnn_v5.yaml --checkpoint V5/checkpoints/run1/best_pq.pt \
    --split test --score-threshold 0.825 --out submissions/v5_run1_best_pq_cutoff0.825.csv
```
