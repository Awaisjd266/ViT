"""Sequential controlled comparisons, seed suites and loss/skip ablations."""
from copy import deepcopy
from pathlib import Path

from crackvit.config import ExperimentConfig
from crackvit.evaluation.runner import evaluate
from crackvit.training import train
from crackvit.utils import write_json


def suite(base: ExperimentConfig, output: Path, models: list[str], seeds: list[int],
          ablations=False) -> list[dict]:
    if not models or not seeds:
        raise ValueError("Models and seeds are required")
    variants = [("bce_dice", 0.5, 0.5, False)]
    if ablations:
        variants += [("bce", 1.0, 0.0, False), ("dice", 0.0, 1.0, False),
                     ("local_skip", 0.5, 0.5, True)]
    results = []
    for name in models:
        for seed in seeds:
            for label, bce, dice, skip in variants:
                if skip and name != "vit":
                    continue
                cfg = deepcopy(base)
                cfg.model.name, cfg.train.seed = name, seed
                cfg.model.local_skip = skip
                cfg.train.bce_weight, cfg.train.dice_weight = bce, dice
                run = output / f"{name}-{label}-seed{seed}"
                cfg.output = str(run)
                training = train(cfg)
                evaluation = evaluate(run / "best.pt", Path(cfg.data.manifest), Path(cfg.data.root),
                                      run / "test", device=cfg.train.device)
                results.append({"model": name, "seed": seed, "variant": label,
                                "training": training, "evaluation": evaluation["metrics"]})
                write_json(output / "suite_results.json", results)
    return results
