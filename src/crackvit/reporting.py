"""Publication-ready plots from executed logs, with visible experiment scope."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_run(run: Path, evaluation: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    history = json.loads((run / "history.json").read_text())
    metrics = json.loads((evaluation / "metrics.json").read_text())
    label = f"Executed {metrics['purpose']} experiment (not manuscript reproduction)"
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 220})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    epochs = [r["epoch"] for r in history]
    for key, name in (("train_loss", "Training"), ("val_loss", "Validation")):
        axes[0].plot(epochs, [r[key] for r in history], marker="o", label=name)
    axes[0].set(xlabel="Epoch", ylabel="BCE–Dice loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [r["val_f1"] for r in history], marker="o", label="Validation F1")
    axes[1].plot(epochs, [r["val_miou"] for r in history], marker="s", label="Validation mIoU")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend(frameon=False)
    fig.suptitle(label)
    fig.savefig(output / "learning_curves.png")
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)
    m = metrics["metrics"]
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=np.float64)
    normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(matrix[i,j]):,}\n{normalized[i,j]:.3f}", ha="center", va="center",
                    color="white" if normalized[i,j] > 0.5 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Background", "Crack"],
           yticklabels=["Background", "Crack"], xlabel="Predicted", ylabel="Reference",
           title=f"Executed {metrics['purpose']} test confusion")
    fig.savefig(output / "confusion_matrix.png")
    fig.savefig(output / "confusion_matrix.pdf")
    plt.close(fig)
    with (output / "executed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["origin", "split", "images", *m])
        writer.writeheader()
        writer.writerow({"origin": f"executed_{metrics['purpose']}", "split": metrics["split"],
                         "images": metrics["images"], **m})
