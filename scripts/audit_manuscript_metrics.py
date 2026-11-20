"""Conditional numerical consistency audit; never changes published observations."""
import csv
import json
from pathlib import Path


def audit(directory: Path, output: Path) -> list[dict]:
    results = []
    for filename in ("manuscript_table_2.csv", "manuscript_table_3.csv"):
        with (directory / filename).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                values = {k: float(row[k].rstrip("%")) / 100
                          for k in ("Precision", "Recall", "F1-score", "Accuracy", "mIoU")}
                p, r, acc = (values[k] for k in ("Precision", "Recall", "Accuracy"))
                f1 = 2 * p * r / (p + r)
                fp, fn = (1 - p) / p, (1 - r) / r
                total = (fp + fn) / (1 - acc)
                tn = total - 1 - fp - fn
                fg_iou = 1 / (1 + fp + fn)
                bg_iou = tn / (tn + fp + fn) if tn >= 0 else None
                implied_miou = (fg_iou + bg_iou) / 2 if bg_iou is not None else None
                results.append({"source": filename, "model": row["Model"],
                                "reported_f1": values["F1-score"], "f1_from_PR": f1,
                                "reported_miou": values["mIoU"],
                                "miou_implied_if_all_metrics_are_micro": implied_miou,
                                "foreground_fraction_implied_if_micro": (1 + fn) / total,
                                "caution": "Conditional on global confusion aggregation; macro averaging can break these identities. Raw predictions are unavailable."})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    audit(Path("docs/manuscript"), Path("reports/manuscript_metric_audit.json"))
