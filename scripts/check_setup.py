"""Sanity check for the GPU / environment."""
import sys

import torch
import torchvision


def main():
    print(f"Python:      {sys.version.split()[0]}")
    print(f"torch:       {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"CUDA:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("GPU:         Apple MPS")


if __name__ == "__main__":
    main()
