"""Plot only measurements exported by executed smoke experiments."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    source = Path("reports/executed/deepcrack/executed_results.csv")
    with source.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    names = [r["model"].upper() for r in rows]
    for ax, metric, title in zip(axes, ["f1", "accuracy"], ["Foreground F1", "Pixel accuracy"]):
        values = [float(r[metric]) for r in rows]
        ax.bar(names, values, color="#377ca8")
        ax.set(ylim=(0, 1), title=title, ylabel="Score")
        for i, v in enumerate(values):
            ax.text(i, v + .02, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("DeepCrack execution checks: 3 epochs, 24 training images, 96×128\nNot a benchmark reproduction or architecture ranking", fontsize=12)
    output = Path("reports/executed/deepcrack")
    fig.savefig(output / "smoke_comparison.png", dpi=220)
    fig.savefig(output / "smoke_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
