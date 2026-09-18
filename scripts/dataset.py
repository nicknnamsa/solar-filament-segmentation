"""COCO-format Dataset for the MAGFiLO filament annotations."""
from torch.utils.data import Dataset


class FilamentDataset(Dataset):
    def __init__(self, image_dir, annotation_file, image_ids=None, transforms=None):
        # TODO: load COCO annotations, filter to image_ids (from splits/*.txt)
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO: return (image tensor, target dict with boxes, labels, masks)
        raise NotImplementedError
