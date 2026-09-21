"""COCO-format Dataset for the MAGFiLO filament annotations."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def read_split(path):
    """Read image ids (one per line) from a split file. Ids are strings like '040301-20140609195854Bh'."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_image_tensor(image_dir, file_name):
    """Load an image as a (3, H, W) float tensor in [0, 1]. Processed copies are PNGs with the same stem."""
    path = Path(image_dir) / file_name
    if not path.exists():
        path = path.with_suffix(".png")
    image = Image.open(path).convert("RGB")
    return torch.as_tensor(np.asarray(image), dtype=torch.float32).permute(2, 0, 1) / 255.0


class FilamentDataset(Dataset):
    """Yields (image, target) in the format torchvision's Mask R-CNN expects.

    Images are grayscale and get replicated to 3 channels to fit the COCO-pretrained backbone.
    Each item is one annotator's version of an image: the same file can appear up to 3 times with
    different masks. All 4 filament categories (Left/Right/Unidentifiable/Ambiguous) map to one class.
    """

    def __init__(self, image_dir, annotation_file, image_ids=None, transforms=None):
        self.image_dir = Path(image_dir)
        self.coco = COCO(str(annotation_file))
        self.image_ids = sorted(image_ids if image_ids is not None else self.coco.getImgIds())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.coco.loadImgs([image_id])[0]
        image = load_image_tensor(self.image_dir, info["file_name"])

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=False))
        masks = [self.coco.annToMask(a) for a in anns]
        masks = [m for m in masks if m.any()]

        h, w = info["height"], info["width"]
        if masks:
            masks = np.stack(masks)
            boxes = []
            for m in masks:
                ys, xs = np.where(m)
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": masks,
            "image_id": image_id,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
