"""Held-out evaluation and validation-only threshold selection."""
import csv
import time
from pathlib import Path

import numpy as np
import torch

from crackvit.config import from_dict
from crackvit.data.manifest import audit_manifest, read_manifest
from crackvit.evaluation.metrics import Confusion, bootstrap_confusions
from crackvit.models import build_model
from crackvit.training import load_checkpoint, make_loader
from crackvit.utils import environment, seed_everything, sha256_file, write_json


def restored_model(checkpoint: Path, device="cpu"):
    ck = load_checkpoint(checkpoint, device)
    cfg = from_dict(ck["config"])
    model = build_model(cfg.model).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck, cfg


def evaluate(checkpoint: Path, manifest: Path, root: Path, output: Path, split="test",
             device="cpu", threshold=None, bootstrap=1000, allow_cross_domain=False) -> dict:
    if split not in {"test", "val"}:
        raise ValueError("Evaluation split must be test or val")
    model, ck, cfg = restored_model(checkpoint, device)
    seed_everything(cfg.train.seed, cfg.train.threads)
    cfg.train.device = device
    chosen_threshold = cfg.threshold if threshold is None else threshold
    if not 0 <= chosen_threshold <= 1:
        raise ValueError("Threshold must lie in [0,1]")
    audit = audit_manifest(manifest, root)
    if not audit["passed"]:
        raise ValueError("Evaluation dataset audit failed")
    records = {r["sample_id"]: r for r in audit["records"]}
    samples = [s for s in read_manifest(manifest) if s.split == split]
    training = [r for r in ck["data_identity"] if r["split"] == "train"]
    train_hashes = {r["image_sha256"] for r in training}
    decoded = {r["decoded_sha256"] for r in training if "decoded_sha256" in r}
    if any(records[s.sample_id]["image_sha256"] in train_hashes or
           records[s.sample_id]["decoded_sha256"] in decoded for s in samples):
        raise ValueError("Evaluation contains an image used for training")
    training_datasets = {r["sample_id"].split("/")[0] for r in training}
    cross_domain = any(s.dataset not in training_datasets for s in samples)
    if cross_domain and not allow_cross_domain:
        raise ValueError("Cross-domain evaluation requires --allow-cross-domain")
    loader = make_loader(samples, root, cfg, ck["stats"])
    confusion, rows, brier_sum, pixels = Confusion(), [], 0.0, 0
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch in loader:
            x, y = batch["image"].to(device), batch["mask"]
            probabilities = model(x).sigmoid().cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Nonfinite model probabilities during evaluation")
            brier_sum += float(((probabilities - y) ** 2).sum())
            pixels += y.numel()
            for sid, p, target in zip(batch["sample_id"], probabilities, y):
                counts = Confusion()
                counts.update(p >= chosen_threshold, target > 0.5)
                confusion.add(counts)
                rec = records[sid]
                rows.append({"sample_id": sid, "group": rec["group"],
                             "image_sha256": rec["image_sha256"], "mask_sha256": rec["mask_sha256"],
                             **counts.metrics()})
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "per_image.json", rows)
    result = {"metrics": confusion.metrics(), "brier_score": brier_sum / pixels,
              "image_bootstrap": bootstrap_confusions(rows, bootstrap, cfg.train.seed),
              "scene_hint_bootstrap": bootstrap_confusions(rows, bootstrap, cfg.train.seed, "group"),
              "threshold": chosen_threshold, "split": split, "images": len(rows),
              "checkpoint_sha256": sha256_file(checkpoint),
              "manifest_sha256": sha256_file(manifest), "purpose": cfg.purpose,
              "cross_domain": cross_domain, "resolution": [cfg.data.height, cfg.data.width],
              "environment": environment(), "benchmark_reproduction": False,
              "aggregation": "dataset-global pixel confusion; absent-class ratios are null; mIoU averages defined classes",
              "limitations": audit["warnings"]}
    write_json(output / "metrics.json", result)
    return result


def select_threshold(checkpoint: Path, manifest: Path, root: Path, output: Path,
                     candidates=None, device="cpu") -> dict:
    model, ck, cfg = restored_model(checkpoint, device)
    seed_everything(cfg.train.seed, cfg.train.threads)
    cfg.train.device = device
    if sha256_file(manifest) != ck["stats"]["manifest_sha256"]:
        raise ValueError("Threshold selection requires the original validation manifest")
    audit = audit_manifest(manifest, root)
    if not audit["passed"]:
        raise ValueError("Threshold data audit failed")
    recorded = {r["sample_id"]: r for r in ck["data_identity"]}
    for row in audit["records"]:
        if row["split"] == "val" and any(row[k] != recorded[row["sample_id"]][k]
                                         for k in ("image_sha256", "mask_sha256")):
            raise ValueError("Validation data changed since training")
    samples = [s for s in read_manifest(manifest) if s.split == "val"]
    candidates = list(np.linspace(0.1, 0.9, 17) if candidates is None else candidates)
    if not candidates or any(not 0 <= t <= 1 for t in candidates):
        raise ValueError("Invalid threshold candidates")
    counts = [Confusion() for _ in candidates]
    with torch.inference_mode():
        for batch in make_loader(samples, root, cfg, ck["stats"]):
            probabilities = model(batch["image"].to(device)).sigmoid().cpu()
            for t, c in zip(candidates, counts):
                c.update(probabilities >= t, batch["mask"] > 0.5)
    results = [{"threshold": float(t), "f1": c.metrics()["f1"]} for t, c in zip(candidates, counts)]
    best = max(results, key=lambda r: r["f1"] if r["f1"] is not None else -1)
    result = {"selected": best, "candidates": results, "selection_split": "val",
              "images": len(samples), "checkpoint_sha256": sha256_file(checkpoint)}
    write_json(output, result)
    return result


def benchmark(checkpoint: Path, output: Path, device="cpu", warmup=5, iterations=30) -> dict:
    if warmup < 0 or iterations < 2:
        raise ValueError("Invalid timing repetitions")
    model, _, cfg = restored_model(checkpoint, device)
    seed_everything(cfg.train.seed, cfg.train.threads)
    x = torch.zeros(1, 3, cfg.data.height, cfg.data.width, device=device)
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    durations = []
    with torch.inference_mode():
        for i in range(warmup + iterations):
            sync()
            start = time.perf_counter()
            model(x)
            sync()
            duration = time.perf_counter() - start
            if i >= warmup:
                durations.append(duration)
    result = {"scope": "model-only batch1 constant-tensor workload; excludes I/O and preprocessing",
              "parameters": sum(p.numel() for p in model.parameters()),
              "median_latency_ms": float(np.median(durations) * 1000),
              "p95_latency_ms": float(np.quantile(durations, 0.95) * 1000),
              "fps_mean": iterations / sum(durations), "warmup": warmup,
              "iterations": iterations, "resolution": [cfg.data.height, cfg.data.width],
              "environment": environment(), "checkpoint_sha256": sha256_file(checkpoint)}
    write_json(output, result)
    return result
