"""Dataset for EdgeAttNet: same underlying data and philosophy as dataset.FilamentDataset (one annotator per
epoch, agreement filtering, augmentation), but the target is a single binary "is this pixel part of any
filament" mask instead of per-instance masks/boxes -- EdgeAttNet is semantic segmentation, not instance
segmentation.
"""
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from dataset import augment, load_image_uint8


class FilamentSemanticDataset(Dataset):
    """Yields (image, binary_mask), both (1, H, W) float tensors in [0, 1]. See FilamentDataset for the shared
    design decisions (agreement filter, one-annotator-per-epoch sampling); the same reasoning applies here."""

    def __init__(self, image_dir, annotation_file, image_ids=None, augment=None, sample_one_annotator=False,
                 agreement=None, resize=None):
        self.image_dir = Path(image_dir)
        self.coco = COCO(str(annotation_file))
        self.image_ids = sorted(image_ids if image_ids is not None else self.coco.getImgIds())
        self.augment_cfg = augment or {}
        self.sample_one_annotator = sample_one_annotator
        self.resize = resize   # int or None; must stay divisible by 16 for EdgeAttNet. Mainly a local-testing
                               # / fallback knob -- the real run trains at native 2048px unless the GPU probe
                               # shows that's impractical (see EdgeAttNet/README.md).

        self.by_file = defaultdict(list)
        for i in self.image_ids:
            self.by_file[self.coco.imgs[i]["file_name"]].append(i)
        self.files = sorted(self.by_file)

        self.keep = self._agreement_flags(agreement["iou"]) if agreement else None

    def _agreement_flags(self, iou_threshold):
        """Identical logic to FilamentDataset._agreement_flags: True if another annotator of the same image
        drew an overlapping filament (IoU >= threshold), or if the image has only one annotator."""
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
        image = load_image_uint8(self.image_dir, info["file_name"]).float() / 255.0   # (1, H, W), our files are
        assert image.shape[0] == 1, "EdgeAttNet expects single-channel grayscale input"               # grayscale

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=False))
        if self.keep is not None:
            anns = [a for a in anns if self.keep[a["id"]]]
        h, w = info["height"], info["width"]
        binary = np.zeros((h, w), dtype=np.uint8)
        for a in anns:
            m = self.coco.annToMask(a)
            binary |= m
        mask = torch.as_tensor(binary, dtype=torch.uint8).unsqueeze(0)   # (1, H, W)

        if self.augment_cfg:
            image, mask = augment(image, mask, self.augment_cfg)
        mask = mask.float()
        if self.resize:
            image = F.interpolate(image.unsqueeze(0), size=(self.resize, self.resize), mode="bilinear",
                                  align_corners=False).squeeze(0)
            # nearest keeps the mask exactly binary; area/bilinear would blur 0/1 into fractional values
            mask = F.interpolate(mask.unsqueeze(0), size=(self.resize, self.resize), mode="nearest").squeeze(0)
        return image, mask


def collate_fn(batch):
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)
