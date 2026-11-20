"""Run authentic-data execution checks and export honest, compact evidence.

No synthetic research images or benchmark targets are generated here.
Run from the repository root after acquisition/preparation.
"""
import argparse
import csv
import json
import logging
from copy import deepcopy
from pathlib import Path

from crackvit.config import load_config
from crackvit.evaluation.runner import benchmark, evaluate, select_threshold
from crackvit.inference import infer
from crackvit.reporting import plot_run
from crackvit.training import train
from crackvit.utils import environment, write_json


def verify(models: list[str], output: Path, dataset: str, seeds: list[int], run_root: Path) -> list[dict]:
    base = load_config("configs/smoke.yaml")
    if dataset == "crack500-patches":
        base.data.manifest = "datasets/manifests/crack500-patches.csv"
        base.data.root = "datasets/raw/crack500"
    records = []
    output.mkdir(parents=True, exist_ok=True)
    for name in models:
        for seed in seeds:
            cfg = deepcopy(base)
            cfg.model.name, cfg.train.seed = name, seed
            run = run_root / dataset / f"{name}-seed{seed}"
            cfg.output = str(run)
            if (run / "summary.json").exists():
                summary = json.loads((run / "summary.json").read_text())
            else:
                summary = train(cfg)
            report = output / f"{name}-seed{seed}"
            result = evaluate(run / "best.pt", Path(cfg.data.manifest), Path(cfg.data.root),
                              report / "test", bootstrap=200)
            timing = benchmark(run / "best.pt", report / "timing.json", warmup=2, iterations=10)
            plot_run(run, report / "test", report / "figures")
            write_json(report / "training_summary.json", summary)
            write_json(report / "config.json", cfg.as_dict())
            write_json(report / "history.json", json.loads((run / "history.json").read_text()))
            write_json(report / "environment.json", json.loads((run / "environment.json").read_text()))
            if name == "vit" and seed == seeds[0]:
                select_threshold(run / "best.pt", Path(cfg.data.manifest), Path(cfg.data.root),
                                 report / "validation_threshold.json")
                import csv as csv_module
                with Path(cfg.data.manifest).open(encoding="utf-8") as f:
                    sample = next(r for r in csv_module.DictReader(f) if r["split"] == "test")
                infer(run / "best.pt", Path(cfg.data.root) / sample["image"],
                      report / "example_inference", tile=128, overlap=32)
            records.append({"dataset": dataset, "model": name, "seed": seed,
                            "origin": "executed_smoke_not_benchmark_reproduction",
                            "train_images": summary["training_samples"],
                            "val_images": summary["validation_samples"],
                            "test_images": result["images"], "epochs": summary["epochs_executed"],
                            "f1": result["metrics"]["f1"], "miou": result["metrics"]["miou"],
                            "accuracy": result["metrics"]["accuracy"],
                            "parameters": summary["parameters"], "fps_cpu_model_only": timing["fps_mean"]})
            write_json(output / "executed_results.json", records)
            with (output / "executed_results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    write_json(output / "execution_environment.json", environment())
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["vit", "cnn", "resnet", "fcn", "unet"])
    p.add_argument("--dataset", choices=["deepcrack", "crack500-patches"], default="deepcrack")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--output", type=Path, default=Path("reports/executed/deepcrack"))
    p.add_argument("--run-root", type=Path, default=Path("runs/verification"))
    a = p.parse_args()
    verify(a.models, a.output, a.dataset, a.seeds, a.run_root)
