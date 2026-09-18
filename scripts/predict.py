"""Run inference on the test images and write an RLE submission CSV."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maskrcnn_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    # TODO: predict masks per test image, RLE-encode, write outputs/submissions/*.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
