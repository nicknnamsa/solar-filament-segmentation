# Solar Filament Segmentation

Instance segmentation of solar filaments in H-Alpha telescope images, for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) on Kaggle.

## Status
🚧 In progress — building baseline model

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

- `data/` — MAGFiLO dataset (COCO-format annotations) and train/val splits
- `scripts/` — data loading, training, evaluation, and prediction scripts
- `notebooks/` — end-to-end pipeline walkthrough
- `outputs/` — model checkpoints and generated submissions

## Usage

```bash
python scripts/check_setup.py       # confirm GPU/environment is working
python scripts/train.py             # fine-tune Mask R-CNN
python scripts/evaluate.py          # score against held-out validation split
python scripts/predict.py           # generate submission CSV
```

## Notes

Dataset is licensed CC BY-NC 4.0 (non-commercial use only).