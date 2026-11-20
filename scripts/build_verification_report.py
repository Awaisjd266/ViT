"""Summarize existing executed evidence without inventing unavailable results."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from crackvit.utils import environment, sha256_file, write_json


def summarize():
    reports, datasets = Path("reports"), Path("datasets")
    audits = {}
    for name in ["deepcrack", "crack500-patches"]:
        a = json.loads((reports / f"{name}_audit.json").read_text())
        audits[name] = {"counts": a["counts"], "passed": a["passed"],
                        "decoded_duplicate_groups": len(a["duplicates"]),
                        "warnings": a["warnings"], "manifest_sha256": a["manifest_sha256"]}
    sources = {
        "deepcrack": {"url": "https://raw.githubusercontent.com/yhlleo/DeepCrack/master/dataset/DeepCrack.zip",
                      "archive_sha256": sha256_file(datasets / "downloads/DeepCrack.zip"),
                      "archive_bytes": (datasets / "downloads/DeepCrack.zip").stat().st_size,
                      "terms": "Noncommercial research and education; original rights retained",
                      "raw_redistribution_authorized": False},
        "crack500": {"source": "https://github.com/fyangneil/pavement-crack-detection",
                     "google_drive_id": "13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD",
                     "archive_sha256": sha256_file(datasets / "downloads/pavement-crack-datasets.zip"),
                     "archive_bytes": (datasets / "downloads/pavement-crack-datasets.zip").stat().st_size,
                     "original_training_archive_present": False,
                     "raw_redistribution_authorized": False}}
    write_json(datasets / "sources.json", sources)
    tree = ET.parse(reports / "test-results.xml")
    suites = list(tree.getroot().iter("testsuite"))
    testing = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    results = []
    for p in sorted((reports / "executed").glob("*/executed_results.json")):
        results.extend(json.loads(p.read_text()))
    expected_suite = Path("runs/pinned-ablations-smoke/suite_results.json")
    ablations = json.loads(expected_suite.read_text()) if expected_suite.exists() else []
    write_json(reports / "executed/ablation_results.json", ablations)
    selection_path = Path("runs/pinned-tuning-smoke/selection.json")
    tuning = json.loads(selection_path.read_text()) if selection_path.exists() else None
    write_json(reports / "executed/tuning_results.json", tuning)
    result = {"date": "2026-11-20", "status": "local implementation verified; full benchmark and remote deployment incomplete",
              "manuscript_sha256": "99eccf2a6e8a70f92da8540d8d4d088d6a30d67baea64b2db868371b15cb4fee",
              "environment": environment(), "tests": testing, "data_audits": audits,
              "executed_smoke_results": results, "ablation_experiments_executed": len(ablations),
              "tuning_trials_executed": len(tuning["trials"]) if tuning else 0,
              "full_manuscript_reproduction_completed": False,
              "raw_dataset_publication_completed": False,
              "outstanding": ["GitHub account with repository write access is required for deployment",
                              "Crack500 original training images missing from inspected official archive",
                              "Full-resolution, full-training-set, multi-seed benchmark experiments remain unexecuted",
                              "Raw dataset redistribution rights require confirmation",
                              "Inferred scene overlaps in official splits need further scientific assessment"]}
    write_json(reports / "verification.json", result)
    return result


if __name__ == "__main__":
    summarize()
