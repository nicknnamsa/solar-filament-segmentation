# Solar Filament Segmentation

Instance segmentation of solar filaments in H-Alpha telescope images, for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) on Kaggle.

## Status
Run 1 (baseline) is finished: PQ 38.6% on the held-out validation images. Run 2 (augmentation, cleaner labels, PQ-based
checkpoint selection) reached 39.6% to 40.2% on the same images, a small gain within the noise. Run 3 (a wider
"ignore margin" around near-miss regions) scored slightly lower, 38.6% to 39.4%. Run 4 (V2's recipe, consolidated
and re-run for 30 epochs) reached 40.5%. Run 5 (a recall-biased Tversky mask loss, motivated by a direct
diagnosis of V4's misses) is the current best at 40.92%, and is simpler than combining with TTA (which adds
nothing further on top of it). See `V1/README.md` through `V5/README.md`.

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
- `notebooks/` — `model_explorer.ipynb` (inspect predictions, tune the score cutoff), `preprocessing_playground.ipynb`, the organisers' `self-evaluation-notebook.ipynb`
- `V1/` — everything from run 1: configs, checkpoints, logs, results (see `V1/README.md`)
- `V2/` — run 2: config and, once trained, checkpoints and logs
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