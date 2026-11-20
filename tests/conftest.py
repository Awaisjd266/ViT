"""Small deterministic synthetic fixtures are tests only, never benchmark data."""

import numpy as np
import pytest
from PIL import Image

from crackvit.data.manifest import Sample, write_manifest


@pytest.fixture
def dataset_fixture(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    samples = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        rgb = np.random.default_rng(i).integers(0, 255, (32, 40, 3), dtype=np.uint8)
        mask = np.zeros((32, 40), np.uint8)
        mask[:, 5 + i:7 + i] = 255
        Image.fromarray(rgb).save(root / f"{i}.png")
        Image.fromarray(mask).save(root / f"{i}_mask.png")
        samples.append(Sample(f"fixture/{i}", "fixture", split, f"{i}.png", f"{i}_mask.png", str(i)))
    manifest = tmp_path / "manifest.csv"
    write_manifest(samples, manifest)
    return root, manifest, samples
