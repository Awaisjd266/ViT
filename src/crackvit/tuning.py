"""Small explicit learning-rate search using validation only."""
from copy import deepcopy
from pathlib import Path

from crackvit.config import ExperimentConfig
from crackvit.training import train
from crackvit.utils import write_json


def tune(base: ExperimentConfig, output: Path, learning_rates: list[float]) -> dict:
    if not learning_rates or any(lr <= 0 for lr in learning_rates):
        raise ValueError("Positive learning rates required")
    trials = []
    for i, lr in enumerate(learning_rates):
        cfg = deepcopy(base)
        cfg.train.learning_rate = lr
        cfg.output = str(output / f"trial-{i:02d}")
        summary = train(cfg)
        trials.append({"learning_rate": lr, "config": cfg.as_dict(), "training": summary})
        write_json(output / "trials.json", trials)
    best = max(trials, key=lambda t: t["training"]["best_validation_f1"])
    result = {"selection": "maximum validation F1; no test evaluation in tuning",
              "trials": trials, "best": best,
              "limitation": "Repeated tuning consumes validation information; freeze search before final test."}
    write_json(output / "selection.json", result)
    return result
