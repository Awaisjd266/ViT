import random
import shutil
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from PIL import Image

from crackvit.data.acquire import safe_extract
from crackvit.data.dataset import CrackDataset, training_statistics
from crackvit.data.manifest import audit_manifest, read_manifest, resolve_under


def test_pair_audit_and_exact_counts(dataset_fixture):
    root, manifest, samples = dataset_fixture
    a = audit_manifest(manifest, root)
    assert a["passed"] and a["counts"] == {"train": 2, "val": 1, "test": 1}
    assert read_manifest(manifest) == samples


def test_decoded_duplicate_leakage(dataset_fixture):
    root, manifest, _ = dataset_fixture
    shutil.copyfile(root / "0.png", root / "3.png")
    a = audit_manifest(manifest, root)
    assert not a["passed"]
    assert a["errors"][-1].get("cross_split_decoded_duplicates")


def test_mask_corruption_rejected(dataset_fixture):
    root, manifest, _ = dataset_fixture
    Image.fromarray(np.full((32, 40), 64, np.uint8)).save(root / "0_mask.png")
    assert not audit_manifest(manifest, root)["passed"]


def test_missing_mask_rejected(dataset_fixture):
    root, manifest, _ = dataset_fixture
    (root / "0_mask.png").unlink()
    result = audit_manifest(manifest, root)
    assert not result["passed"] and result["errors"][0]["sample"] == "fixture/0"


def test_spatial_mismatch_rejected(dataset_fixture):
    root, manifest, _ = dataset_fixture
    Image.new("L", (2, 2)).save(root / "0_mask.png")
    assert not audit_manifest(manifest, root)["passed"]


def test_training_statistics_do_not_use_test(dataset_fixture):
    root, _, samples = dataset_fixture
    before = training_statistics(samples, root, 32, 40)
    Image.new("RGB", (40, 32), "white").save(root / "3.png")
    after = training_statistics(samples, root, 32, 40)
    assert before == after and before["training_images"] == 2


def test_augmentation_retains_binary_aligned_mask(dataset_fixture):
    root, _, samples = dataset_fixture
    mask = np.asarray(Image.open(root / "0_mask.png"))
    Image.fromarray(np.repeat(mask[..., None], 3, -1)).save(root / "0.png")
    dataset = CrackDataset(samples[:1], root, 32, 40, {"mean": [0, 0, 0], "std": [1, 1, 1]}, True)
    random.seed(11)
    sample = dataset[0]
    assert set(sample["mask"].unique().tolist()) == {0, 1}
    assert torch.equal(sample["image"][0] > .5, sample["mask"][0] > .5)


def test_path_escape_and_zip_slip(tmp_path):
    with pytest.raises(ValueError):
        resolve_under(tmp_path, "../secret")
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("../evil.txt", "bad")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "evil.txt").exists()


def test_archive_size_limit(tmp_path):
    archive = tmp_path / "big.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("large", "x" * 100)
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "extract", max_bytes=50)
