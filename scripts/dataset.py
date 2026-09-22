"""COCO-format Dataset for the MAGFiLO filament annotations."""
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
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


def load_image_uint8(image_dir, file_name):
    """Like load_image_tensor but as uint8 and, for grayscale files, a single channel: about 12x less data to move
    between processes. The training loop converts to float and expands to 3 channels on the GPU."""
    path = Path(image_dir) / file_name
    if not path.exists():
        path = path.with_suffix(".png")
    image = Image.open(path)
    if image.mode != "L":
        image = image.convert("RGB")
    array = np.asarray(image)
    return torch.from_numpy(array.copy()).unsqueeze(0) if array.ndim == 2 else torch.from_numpy(array.copy()).permute(2, 0, 1)


def apply_jitter(image, cfg):
    """Random brightness shift and contrast gain on a float image in [0, 1]."""
    shift = random.uniform(-cfg.get("brightness", 0), cfg.get("brightness", 0))
    gain = 1 + random.uniform(-cfg.get("contrast", 0), cfg.get("contrast", 0))
    return ((image - 0.5) * gain + 0.5 + shift).clamp(0, 1)


def boxes_from_masks(masks):
    """(N, H, W) masks -> (N, 4) float boxes [x0, y0, x1, y1]."""
    rows, cols = masks.any(dim=2), masks.any(dim=1)
    boxes = []
    for r, c in zip(rows, cols):
        ys, xs = torch.nonzero(r).squeeze(1), torch.nonzero(c).squeeze(1)
        boxes.append([xs[0].item(), ys[0].item(), xs[-1].item() + 1, ys[-1].item() + 1])
    return torch.as_tensor(boxes, dtype=torch.float32)


def augment(image, masks, cfg):
    """Random flips / 90-degree rotations / brightness and contrast jitter. The sun has no fixed 'up', so these are safe.

    cfg keys (all optional): flip (bool), rot90 (bool), brightness (max shift), contrast (max relative change).
    """
    if cfg.get("flip"):
        if random.random() < 0.5:
            image, masks = image.flip(-1), masks.flip(-1)
        if random.random() < 0.5:
            image, masks = image.flip(-2), masks.flip(-2)
    if cfg.get("rot90"):
        k = random.randint(0, 3)
        image, masks = torch.rot90(image, k, (1, 2)), torch.rot90(masks, k, (1, 2))
    if (cfg.get("brightness") or cfg.get("contrast")) and image.dtype.is_floating_point:
        image = apply_jitter(image, cfg)             # uint8 images get their jitter on the GPU (see train.py)
    return image, masks


class FilamentDataset(Dataset):
    """Yields (image, target) in the format torchvision's Mask R-CNN expects.

    Images are grayscale and get replicated to 3 channels to fit the COCO-pretrained backbone.
    Each image id is one annotator's version of an image: the same file can appear up to 3 times with
    different masks. All 4 filament categories (Left/Right/Unidentifiable/Ambiguous) map to one class.

    Optional training behaviours (all off by default, which reproduces the first training run):
      augment:              dict for `augment` above.
      sample_one_annotator: one item per FILE; each time it is loaded, a random annotator's version is used.
      uint8_images:         return 8-bit (1-channel for grayscale) images; the training loop finishes the conversion
                            on the GPU. Much less data to move between loader processes.
      agreement:            {"iou": x}. Drop filaments that no other annotator of the same image drew (IoU >= x).
                            Images with a single annotator keep everything. Under the scoring, a filament only one
                            annotator drew is a false alarm against the others, so the model should not learn it.
    """

    def __init__(self, image_dir, annotation_file, image_ids=None, augment=None, sample_one_annotator=False,
                 agreement=None, uint8_images=False):
        self.image_dir = Path(image_dir)
        self.uint8_images = uint8_images
        self.coco = COCO(str(annotation_file))
        self.image_ids = sorted(image_ids if image_ids is not None else self.coco.getImgIds())
        self.augment = augment or {}
        self.sample_one_annotator = sample_one_annotator

        self.by_file = defaultdict(list)
        for i in self.image_ids:
            self.by_file[self.coco.imgs[i]["file_name"]].append(i)
        self.files = sorted(self.by_file)

        self.keep = self._agreement_flags(agreement["iou"]) if agreement else None
        if self.keep is not None:
            total = sum(len(self.coco.imgToAnns.get(i, [])) for i in self.image_ids)
            dropped = sum(1 for v in self.keep.values() if not v)
            print(f"agreement filter (IoU >= {agreement['iou']}): dropped {dropped} of {total} filaments "
                  f"({100 * dropped / max(total, 1):.1f}%) that no other annotator drew")

    def _agreement_flags(self, iou_threshold):
        """{annotation id: True if another annotator of the same image drew an overlapping filament, or if nobody else annotated it}."""
        flags = {}
        for ids in self.by_file.values():
            anns = {i: self.coco.imgToAnns.get(i, []) for i in ids}
            for i in ids:
                for a in anns[i]:
                    flags[a["id"]] = len(ids) < 2
            if len(ids) < 2:
                continue
            rles = {i: [self.coco.annToRLE(a) for a in anns[i]] for i in ids}
            for i in ids:
                others = [r for j in ids if j != i for r in rles[j]]
                if not others or not rles[i]:
                    continue
                best = mask_util.iou(rles[i], others, [0] * len(others)).max(axis=1)
                for a, b in zip(anns[i], best):
                    flags[a["id"]] = bool(b >= iou_threshold)
        return flags

    def __len__(self):
        return len(self.files) if self.sample_one_annotator else len(self.image_ids)

    def __getitem__(self, idx):
        if self.sample_one_annotator:
            image_id = random.choice(self.by_file[self.files[idx]])
        else:
            image_id = self.image_ids[idx]
        info = self.coco.loadImgs([image_id])[0]
        load = load_image_uint8 if self.uint8_images else load_image_tensor
        image = load(self.image_dir, info["file_name"])

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=False))
        if self.keep is not None:
            anns = [a for a in anns if self.keep[a["id"]]]
        masks = [self.coco.annToMask(a) for a in anns]
        masks = [m for m in masks if m.any()]

        h, w = info["height"], info["width"]
        if masks:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            if self.augment:
                image, masks = augment(image, masks, self.augment)
            boxes = boxes_from_masks(masks.bool())
        else:
            if self.augment:
                image, _ = augment(image, torch.zeros((0, h, w), dtype=torch.uint8), self.augment)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": masks,
            "image_id": image_id,
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
