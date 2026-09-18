"""Score a checkpoint on the local validation split (Dice / Panoptic Quality)."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maskrcnn_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    # TODO: run inference on val split, compute Dice and PQ
    raise NotImplementedError


if __name__ == "__main__":
    main()
