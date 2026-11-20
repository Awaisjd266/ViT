"""Exact binary confusion counts and explicitly defined aggregation."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, prediction, target) -> None:
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        p, y = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
        if p.shape != y.shape:
            raise ValueError("Prediction and target shapes differ")
        self.tp += int(np.count_nonzero(p & y))
        self.fp += int(np.count_nonzero(p & ~y))
        self.fn += int(np.count_nonzero(~p & y))
        self.tn += int(np.count_nonzero(~p & ~y))

    def add(self, other: "Confusion") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    def counts(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}

    def metrics(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        n = tp + fp + fn + tn
        if n == 0:
            raise ValueError("Cannot evaluate zero pixels")
        def ratio(a, b):
            return a / b if b else None
        fg = ratio(tp, tp + fp + fn)
        bg = ratio(tn, tn + fp + fn)
        present = [x for x in (fg, bg) if x is not None]
        return {**self.counts(), "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
                "f1": ratio(2 * tp, 2 * tp + fp + fn), "crack_iou": fg,
                "background_iou": bg, "miou": sum(present) / len(present),
                "accuracy": (tp + tn) / n, "foreground_fraction": (tp + fn) / n,
                "pixels": n}


def bootstrap_confusions(records: list[dict], iterations=1000, seed=42,
                         group_key="sample_id") -> dict:
    """Resample entire image/scene units; pixels are not independent replicates."""
    grouped = {}
    for row in records:
        key = row[group_key]
        grouped.setdefault(key, Confusion()).add(Confusion(**{k: int(row[k]) for k in ("tp", "fp", "fn", "tn")}))
    units = list(grouped.values())
    if len(units) < 2 or iterations < 2:
        return {"unit": group_key, "units": len(units), "intervals": None}
    rng, draws = np.random.default_rng(seed), {k: [] for k in ("f1", "miou", "accuracy")}
    for _ in range(iterations):
        c = Confusion()
        for idx in rng.integers(0, len(units), size=len(units)):
            c.add(units[idx])
        for key in draws:
            value = c.metrics()[key]
            if value is not None:
                draws[key].append(value)
    return {"unit": group_key, "units": len(units), "iterations": iterations, "seed": seed,
            "intervals": {k: np.quantile(v, [0.025, 0.975]).tolist() if v else None
                          for k, v in draws.items()}}


def paired_bootstrap(a: list[dict], b: list[dict], iterations=1000, seed=42) -> dict:
    if iterations < 2:
        raise ValueError("At least two bootstrap iterations required")
    aa, bb = {r["sample_id"]: r for r in a}, {r["sample_id"]: r for r in b}
    if aa.keys() != bb.keys() or len(aa) < 2:
        raise ValueError("Paired comparison needs the same two or more sample IDs")
    keys = sorted(aa)
    if any(any(aa[k][n] != bb[k][n] for n in ("image_sha256", "mask_sha256")) for k in keys):
        raise ValueError("Paired samples have different image/mask identities")
    rng, differences = np.random.default_rng(seed), []
    for _ in range(iterations):
        ca, cb = Confusion(), Confusion()
        for i in rng.integers(0, len(keys), len(keys)):
            for row, c in ((aa[keys[i]], ca), (bb[keys[i]], cb)):
                c.add(Confusion(**{n: int(row[n]) for n in ("tp", "fp", "fn", "tn")}))
        va, vb = ca.metrics()["f1"], cb.metrics()["f1"]
        if va is not None and vb is not None:
            differences.append(va - vb)
    return {"metric": "micro_f1_a_minus_b", "unit": "image", "seed": seed,
            "images": len(keys), "iterations": iterations,
            "ci95": np.quantile(differences, [0.025, 0.975]).tolist() if differences else None,
            "limitation": "Image bootstrap does not establish independent source scenes or account for training-seed variation."}
