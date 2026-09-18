"""Fine-tune Mask R-CNN on the filament training split."""
import argparse

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maskrcnn_baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # TODO: build datasets/loaders, model, optimizer; run training loop; save checkpoints
    raise NotImplementedError


if __name__ == "__main__":
    main()
