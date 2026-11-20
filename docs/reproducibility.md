# Reproducibility and execution protocol

## Rebuild data and environment

1. Create a clean Python3.12 environment and install torch2.6/torchvision0.21 with an appropriate CPU or CUDA index. Install this package plus dev/download extras. The local locked environment is in `requirements-lock.txt`; CUDA builds may require a distinct lock.
2. Read source restrictions in `datasets/DATASET_CARD.md`. Acquire authentic archives with `crackvit download`. Verify checksums against `datasets/sources.json` before reusing the observed release.
3. Prepare DeepCrack using seed42. Official300/237 membership is retained; source-group-aware selection yields240/60 within the official training set. Do not derive validation from test.
4. The official Crack500 bundle supports complete1896/348/1124 crops. Its original training archive is absent. Use the labeled patch configuration or independently obtain original training data and document its source. Do not fabricate missing pixels or use arbitrary internet photographs without valid pixel masks.
5. Audit all pairs before every run. Preserve audit warnings. Exact and decoded duplicates are detected; acquisition-scene independence needs additional evidence. A separate source-disjoint protocol can reduce known grouping risks, with changed counts disclosed.

## Execute planned research

`configs/deepcrack.yaml` and `configs/crack500.yaml` preserve the manuscript's resolutions and intended original-image splits where data are available. ViT dimensions, optimizer, epoch count and decoder details are selected defaults. Document any change in a new configuration and output directory. Run predefined seed42/43/44 suites; run controlled loss and local-skip ablations without test-informed tuning.

Report train/validation/test manifests, all seeds, actual stopping epochs, failures, thresholds, uncertainty and exact current checkpoint hashes. Include full model definitions and source commit. The source manuscript's stated numbers are comparator claims, not success thresholds used to tune tests.

## Run evidence

`scripts/verify_real_data.py` executes a small96×128, three-epoch,24-train/12-validation smoke workload on authentic data. It evaluates all official test samples in the selected protocol, exports per-image counts, writes plots and performs model-only timing. Those observations demonstrate code execution only. Their lower resolution and tiny training subset preclude direct comparison to the original tables.

`reports/test-results.xml` is the actual local unit/integration JUnit report. `reports/verification.json` is a machine-readable status snapshot with executed checks, data availability and unresolved limitations. `reports/executed/` contains actual JSON/CSV plots and environment records. Original claims live separately in `docs/manuscript/`.

## Checkpoints and deterministic resume

All checkpoint writes are atomic; same-config resume restores optimizer, scheduler, scaler and relevant randomness. Config equality and data identities are enforced. Tests compare uninterrupted versus interrupted/resumed worker-zero CPU training. PyTorch deterministic algorithms are enabled. Reproducibility across different accelerators, BLAS/thread configurations and library versions is not guaranteed. Record actual hardware and dependency locks rather than assuming historical environment equivalence.

## Deployment

The initial remote inspection found Awaisjd266/ViT empty, with no branches/files. The connected plugin identity initially was mawaisashraf3 and had no push permission. The repository subsequently gained a calculator; its index.html and original instructions were preserved when merging the research project. After user-completed Git Credential Manager authorization as Awaisjd266, the complete project was pushed to main on 20 November 2026. GitHub's branch API independently verified uploaded commit `016d4d839e836feda5ed2d8c88ea4d2e9598778b`. `reports/deployment.json` records this verified payload commit; later metadata commits may follow it. Raw source datasets are excluded from normal pushes until redistribution rights are established.
