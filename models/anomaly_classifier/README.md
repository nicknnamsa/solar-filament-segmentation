# Anomaly classifier: a side tool, not currently wired into the main pipeline

A small CNN (`scripts/anomaly_classifier.py`) trained on the external "GONG H-Alpha Anomaly Dataset"
(`data/GONG H-Alpha Anomaly Dataset/`) to classify a full-disk H-alpha image as "good" or "anomalous" (real
image-quality defects — not filaments). This dataset has no filament labels at all, so it can't be used as
filament training data directly; it was built as a possible quality-gate tool for future candidate images.

## Result

Strong on its own benchmark (99.2% accuracy, 97.5% precision/recall on its held-out test split), but a real,
honest gap on our own filament-competition images: of the 4 images in our own dataset independently known to be
anomalous, it only correctly flagged 1 (25% recall) — while still correctly passing 112 of 115 known-good images
(97% specificity). n=4 is small enough that this isn't a precise number, but it's a real signal that its own
benchmark performance doesn't fully transfer to our specific images.

**Not currently used for anything** — held back deliberately rather than wired into training or prediction,
pending either (a) fine-tuning on a larger sample of our own labeled good/anomalous images, or (b) treating its
output as a soft secondary signal rather than a hard filter. See the main project's conversation history for the
full reasoning; this was a side investigation, not part of the final submitted pipeline.

## Files

`checkpoints/run1/best_f1.pt`, `history.csv`. Not tracked by git.

```bash
python scripts/anomaly_classifier.py --data "data/GONG H-Alpha Anomaly Dataset"
```
