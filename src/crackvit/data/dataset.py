"""Training-only statistics and synchronized segmentation preprocessing."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from crackvit.data.manifest import Sample, resolve_under


def load_pair(root: Path, sample: Sample) -> tuple[Image.Image, Image.Image]:
    with Image.open(resolve_under(root, sample.image)) as f:
        image = f.convert("RGB")
    with Image.open(resolve_under(root, sample.mask)) as f:
        mask = f.convert("L")
    return image, mask


def training_statistics(samples: list[Sample], root: Path, height: int, width: int) -> dict:
    train = [s for s in samples if s.split == "train"]
    if not train:
        raise ValueError("No training images for statistics")
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    pixels, foreground = 0, 0
    for s in train:
        image, mask = load_pair(root, s)
        a = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float64) / 255
        total += a.sum(axis=(0, 1))
        squares += (a * a).sum(axis=(0, 1))
        pixels += height * width
        foreground += int((np.asarray(mask.resize((width, height), Image.Resampling.NEAREST)) > 0).sum())
    mean = total / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean**2, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "training_images": len(train),
            "pixels": pixels, "foreground_pixels": foreground,
            "foreground_fraction": foreground / pixels, "height": height, "width": width}


def preprocess(image: Image.Image, height: int, width: int, stats: dict) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                     dtype=np.float32) / 255
    arr = (arr - np.asarray(stats["mean"], dtype=np.float32)) / np.maximum(
        np.asarray(stats["std"], dtype=np.float32), 1e-6)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1)


class CrackDataset(Dataset):
    def __init__(self, samples: list[Sample], root: Path, height: int, width: int,
                 stats: dict, augment: bool = False):
        self.samples, self.root = samples, root
        self.height, self.width, self.stats, self.augment = height, width, stats, augment
        if not samples:
            raise ValueError("Dataset subset is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        s = self.samples[index]
        image, mask = load_pair(self.root, s)
        if self.augment:
            # Dihedral transforms do not interpolate thin masks or invent mask values.
            turns = random.randrange(4)
            for _ in range(turns):
                image = image.transpose(Image.Transpose.ROTATE_90)
                mask = mask.transpose(Image.Transpose.ROTATE_90)
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if random.random() < 0.5:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)
        x = preprocess(image, self.height, self.width, self.stats)
        y = np.asarray(mask.resize((self.width, self.height), Image.Resampling.NEAREST)) > 0
        return {"image": x, "mask": torch.from_numpy(y.astype(np.float32)).unsqueeze(0),
                "sample_id": s.sample_id}
