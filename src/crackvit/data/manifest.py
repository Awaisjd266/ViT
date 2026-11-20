"""Portable sample manifests, official split adapters and leakage audits."""
import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from crackvit.utils import sha256_file, write_json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset: str
    split: str
    image: str
    mask: str
    group: str


def resolve_under(root: Path, relative: str) -> Path:
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside dataset root: {relative}")
    return p


def write_manifest(samples: list[Sample], output: Path) -> None:
    if not samples:
        raise ValueError("Empty manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(samples[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in samples)


def read_manifest(path: Path) -> list[Sample]:
    with path.open(newline="", encoding="utf-8") as f:
        samples = [Sample(**row) for row in csv.DictReader(f)]
    if not samples or any(s.split not in SPLITS for s in samples):
        raise ValueError("Manifest is empty or has invalid split values")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    return samples


def pair_folders(root: Path, images: str, masks: str, dataset: str, split: str) -> list[Sample]:
    imfiles = sorted(p for p in (root / images).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    maskfiles = {p.stem: p for p in (root / masks).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    if len(maskfiles) != len([p for p in (root / masks).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]):
        raise ValueError("Ambiguous mask stems")
    samples = []
    used = set()
    for image in imfiles:
        mask = maskfiles.get(image.stem)
        if mask is None:
            raise ValueError(f"Missing mask for {image}")
        used.add(mask.stem)
        # Conservative scene hint, not an assertion of independently acquired scenes.
        group = re.sub(r"[-_]\d+$", "", image.stem)
        samples.append(Sample(f"{dataset}/{images}/{image.stem}", dataset, split,
                              image.relative_to(root).as_posix(), mask.relative_to(root).as_posix(),
                              f"{dataset}/{group}"))
    if set(maskfiles) != used or not samples:
        raise ValueError("Unpaired masks or empty image directory")
    return samples


def prepare_deepcrack(root: Path, output: Path, seed: int = 42) -> list[Sample]:
    train = pair_folders(root, "train_img", "train_lab", "deepcrack", "train")
    test = pair_folders(root, "test_img", "test_lab", "deepcrack", "test")
    if len(train) != 300 or len(test) != 237:
        raise ValueError(f"Official counts expected 300/237, received {len(train)}/{len(test)}")
    # Group inferred source scenes within official training data; official
    # train/test overlaps are reported, never silently reassigned.
    groups = defaultdict(list)
    for s in train:
        with Image.open(root / s.image) as image:
            digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        groups[s.group].append((digest, s))
    parent = {g: g for g in groups}
    def find(g):
        while parent[g] != g:
            g = parent[g]
        return g
    seen = {}
    for g, members in groups.items():
        for digest, _ in members:
            if digest in seen:
                parent[find(g)] = find(seen[digest])
            seen[digest] = g
    merged = defaultdict(list)
    for g, members in groups.items():
        merged[find(g)].extend(s for _, s in members)
    groups = merged
    items = list(sorted(groups.items()))
    random.Random(seed).shuffle(items)
    # Exact subset sum gives 60 validation samples while preserving duplicate groups.
    reachable: dict[int, list[str]] = {0: []}
    for digest, members in items:
        for total, selected in list(reachable.items())[::-1]:
            new = total + len(members)
            if new <= 60 and new not in reachable:
                reachable[new] = selected + [digest]
    if 60 not in reachable:
        raise ValueError("Cannot produce exactly 60 validation images without splitting duplicates")
    chosen = set(reachable[60])
    result = [Sample(s.sample_id, s.dataset, "val" if key in chosen else "train",
                     s.image, s.mask, s.group) for key, members in items for s in members] + test
    result.sort(key=lambda s: s.sample_id)
    write_manifest(result, output)
    return result


def prepare_crack500(root: Path, output: Path) -> list[Sample]:
    """Support canonical original folders; never substitute thousands of filtered crops."""
    result = []
    for folder, split, count in (("train", "train", 250), ("val", "val", 50), ("test", "test", 200)):
        directory = root / folder if (root / folder).exists() else root / f"{folder}data"
        if not directory.exists():
            raise ValueError(f"Missing original-image folder {folder}/ or {folder}data/; crops are not substitutes")
        images = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
        if len(images) != count:
            raise ValueError(f"Expected {count} original {folder} images, got {len(images)}")
        for image in images:
            mask = image.with_name(image.stem + "_mask.png")
            if not mask.exists():
                mask = image.with_suffix(".png")
            if not mask.exists():
                raise ValueError(f"Missing original Crack500 mask: {mask}")
            result.append(Sample(f"crack500/{image.stem}", "crack500", split,
                                 image.relative_to(root).as_posix(), mask.relative_to(root).as_posix(),
                                 f"crack500/{image.stem}"))
    write_manifest(result, output)
    return result


def audit_manifest(manifest: Path, root: Path, output: Path | None = None) -> dict:
    samples = read_manifest(manifest)
    errors, warnings, records = [], [], []
    hashes, groups = defaultdict(list), defaultdict(set)
    for s in samples:
        try:
            ip, mp = resolve_under(root, s.image), resolve_under(root, s.mask)
            with Image.open(ip) as im, Image.open(mp) as ma:
                rgb = np.asarray(im.convert("RGB"))
                raw_mask = np.asarray(ma)
                if raw_mask.ndim == 3:
                    if not np.all(raw_mask == raw_mask[..., :1]):
                        raise ValueError("Mask channels disagree")
                    raw_mask = raw_mask[..., 0]
                if rgb.shape[:2] != raw_mask.shape:
                    raise ValueError("Image-mask spatial mismatch")
                vals = np.unique(raw_mask)
                if not set(vals.tolist()).issubset({0, 1, 255}):
                    raise ValueError(f"Nonbinary mask values: {vals[:12].tolist()}")
                fg = float((raw_mask > 0).mean())
                digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
                hashes[digest].append((s.sample_id, s.split))
                groups[s.group].add(s.split)
                records.append({**asdict(s), "image_sha256": sha256_file(ip),
                                "mask_sha256": sha256_file(mp), "decoded_sha256": digest,
                                "height": rgb.shape[0], "width": rgb.shape[1],
                                "foreground_fraction": fg})
        except (OSError, ValueError) as exc:
            errors.append({"sample": s.sample_id, "error": str(exc)})
    duplicate_groups = [v for v in hashes.values() if len(v) > 1]
    cross_split = [v for v in duplicate_groups if len({s for _, s in v}) > 1]
    if cross_split:
        errors.append({"cross_split_decoded_duplicates": cross_split})
    shared = {g: sorted(v) for g, v in groups.items() if len(v) > 1}
    if shared:
        warnings.append({"inferred_scene_groups_cross_splits": shared,
                         "action": "Official split retained. Independence is not established by image hashes."})
    result = {"manifest_sha256": sha256_file(manifest), "count": len(samples),
              "counts": dict(Counter(s.split for s in samples)), "errors": errors,
              "warnings": warnings, "duplicates": duplicate_groups, "records": records,
              "passed": not errors}
    if output:
        write_json(output, result)
    return result


def prepare_crack500_crops(root: Path, output: Path) -> list[Sample]:
    """Explicit alternative 1896/348/1124 crop protocol, never called 500 originals."""
    result = []
    for folder, split, expected in (("traincrop", "train", 1896),
                                    ("valcrop", "val", 348), ("testcrop", "test", 1124)):
        images = sorted((root / folder).glob("*.jpg"))
        if len(images) != expected:
            raise ValueError(f"Expected {expected} official {folder} patches, got {len(images)}")
        for image in images:
            mask = image.with_suffix(".png")
            if not mask.exists():
                raise ValueError(f"Missing patch mask: {mask}")
            parent = re.sub(r"_\d+_\d+$", "", image.stem)
            result.append(Sample(f"crack500-patches/{folder}/{image.stem}", "crack500-patches", split,
                                 image.relative_to(root).as_posix(), mask.relative_to(root).as_posix(),
                                 f"crack500/{parent}"))
    write_manifest(result, output)
    return result
