# Solar Filament Segmentation

Instance segmentation of solar filaments in H-Alpha telescope images, for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) on Kaggle.

## Status
**Final: V5 (recall-biased Tversky loss) + a merge-radius post-processing step, PQ 41.7% on the held-out
validation set, verified against the official scorer.** See `notebooks/pipeline_walkthrough.ipynb` for the full
narrated summary of the project, or the short version below.

Run 1 (baseline): PQ 38.6%. Run 2 (augmentation, cleaner labels, PQ-based checkpoint selection): 39.6-40.2%. Run 3
(a wider "ignore margin" around near-miss regions) was a regression: 38.6-39.4%, reverted. Run 4 (V2's recipe,
consolidated, 30 epochs): 40.5%. Run 5 (a recall-biased Tversky mask loss): 40.9%, simpler than combining with
TTA (which added nothing further on top of it). Run 6a/6b (a size-weighted loss and an anomaly-image-excluded
training set, both targeting a measured "small/faint filaments get missed entirely" pattern): neither beat V5.
A side investigation into a different architecture (EdgeAttNet, a published edge-attention U-Net, reimplemented
cleanly rather than using the authors' released weights due to a real leakage risk) also underperformed V5, as
did every tested way of ensembling it with V5. The one thing that *did* help: post-training, merging nearby
detections (`predict.merge_nearby_instances`) gave V5 a free +0.8pp, since 78% of its false positives turned out
to be near-duplicates of real filaments rather than pure noise. See `V1/README.md` through `V6/README.md` and
`EdgeAttNet/README.md` for the full detail behind each of these.

## The task

Given a grayscale solar image, identify each individual filament (dense clouds of solar material) and output a separate mask for each one. This is instance segmentation, not simple binary segmentation — multiple filaments can appear in one image and need to be told apart.

Scored using the Panoptic Quality metric, which penalizes both fragmented predictions (one real filament split into several) and over-merged predictions (several real filaments combined into one).

## Approach

Baseline: fine-tuned Mask R-CNN (ResNet-50 backbone, pretrained on COCO) via `torchvision`.

Known hard parts of this dataset, from the competition's own description:
- Thin, thread-like "barb" structures that standard segmentation losses tend to under-weight
- Background noise from ground-based observation that can resemble filament material
- Structural continuity — avoiding fragmented or over-merged predictions

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11+. Training runs on a rented GPU (RunPod); local machine is used for data prep and inference testing.

## Structure

- `data/` — MAGFiLO dataset (COCO-format annotations), train/val splits, and derived images (`processed/`)
- `scripts/` — shared code: data loading, training, evaluation, prediction, scoring (`metrics.py` is the official metric)
- `notebooks/` — `pipeline_walkthrough.ipynb` (the full project summary — start here), `model_explorer.ipynb`
  (inspect predictions, tune the score cutoff), `preprocessing_playground.ipynb`, the organisers'
  `self-evaluation-notebook.ipynb`
- `V1/` through `V6/` — each numbered run: configs, checkpoints, logs, results (see each run's own `README.md`)
- `EdgeAttNet/` — a side investigation into a different, purpose-built architecture; underperformed V5 (see its README)
- `anomaly_classifier/` — a small classifier trained on an external good/anomalous image-quality dataset; not
  currently wired into the main pipeline (see the note in that folder)
- `submissions/` — CSV files to upload to Kaggle, with a log of how each was made
- `outputs/` — exploratory visualisations

Each run folder holds its own `configs/`, `checkpoints/` and `logs/`; the code in `scripts/` is shared, and new training
options are switched on from the config, so older configs keep working.

## Usage

```bash
python scripts/check_setup.py                                             # confirm GPU/environment is working
python scripts/make_splits.py                                             # only needed once; splits are already in data/splits/
python scripts/train.py --config V2/configs/maskrcnn_v2.yaml             # fine-tune Mask R-CNN
python scripts/evaluate.py --checkpoint V2/checkpoints/run1/best_pq.pt   # official PQ on the validation split, several cutoffs
python scripts/predict.py --checkpoint V2/checkpoints/run1/best_pq.pt    # generate submission CSV
python scripts/annotator_agreement.py                                     # how well the annotators agree with each other
```

To use a run-1 model, pass its config as well, e.g. `--config V1/configs/maskrcnn_baseline.yaml --checkpoint V1/checkpoints/raw/last.pt`.

## Notes

Dataset is licensed CC BY-NC 4.0 (non-commercial use only).