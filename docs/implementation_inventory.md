# Implementation inventory

## Source modules

| File | Function and integration |
|---|---|
| `src/crackvit/cli.py` | Validates CLI arguments and dispatches every pipeline command; propagates explicit failures through nonzero exit codes |
| `src/crackvit/__main__.py` | Supports `python -m crackvit` |
| `src/crackvit/config.py` | Typed data/model/training configs, unknown-key validation and parameter checks |
| `src/crackvit/data/acquire.py` | Authenticated-free source downloads, provenance, bounded safe extraction and Crack500 nested archive handling |
| `src/crackvit/data/manifest.py` | Dataset adapters, source-group-aware DeepCrack validation selection, strict source counts and image/mask integrity audits |
| `src/crackvit/data/dataset.py` | Selected-training channel statistics, synchronized orientation augmentation, deterministic resizing and tensor conversion |
| `src/crackvit/models/vit.py` | Global attention encoder, positional interpolation, progressive decoder, optional local feature skip and arbitrary-size padding/cropping |
| `src/crackvit/models/baselines.py` | Explicit CNN, U-Net and ResNet18 dense models |
| `src/crackvit/models/__init__.py` | Single model factory and standard FCN-ResNet50 adapter |
| `src/crackvit/losses.py` | Stable weighted BCEWithLogits plus soft Dice overlap with true targets |
| `src/crackvit/training.py` | Audited loaders, optimization, gradient clipping, optional CUDA AMP, cosine schedule, early stopping, atomic checkpoints and resume |
| `src/crackvit/evaluation/metrics.py` | Exact confusion counts, explicit class conventions, image/group bootstrap and paired comparisons |
| `src/crackvit/evaluation/runner.py` | Frozen held-out evaluation, validation-only threshold search, normalization reuse and synchronized model timing |
| `src/crackvit/inference.py` | Native-resolution tiled probability prediction, cosine blending, masks and overlays |
| `src/crackvit/experiments.py` | Controlled model/seed/loss/skip suites using common config |
| `src/crackvit/tuning.py` | Validation-only learning-rate search; persists trial config and selection |
| `src/crackvit/reporting.py` | Learning curves, normalized confusion plots, vector PDF/PNG and CSV output from actual run records |
| `src/crackvit/storage.py` | SQLite run/epoch schema, foreign keys, parameterized transaction writes and recoverable epoch updates |
| `src/crackvit/utils.py` | Seeds, deterministic settings, SHA256, atomic JSON and measured environment metadata |

Package `__init__.py` files define import boundaries. Every substantive module is used by the CLI, pipeline or verification suite. The system does not need an unrelated service/database layer.

## Scripts and tests

| Component | Purpose |
|---|---|
| `scripts/extract_manuscript_assets.py` | Export explicitly requested original figures/tables with manuscript hash and claim labels |
| `scripts/audit_manuscript_metrics.py` | Conditional arithmetic/aggregation audit of original table claims |
| `scripts/verify_real_data.py` | Execute authentic-data training, test evaluation, timing, threshold selection, inference and plot export |
| `scripts/build_verification_report.py` | Summarize existing artifacts; marks absent original data and deployment as incomplete |
| `scripts/plot_comparison.py` | Visualize actual measured short-run results with a scope label |
| `tests/test_metrics_losses.py` | Known confusion values, absent classes, micro identities, finite gradients and bootstrap checks |
| `tests/test_data.py` | Pairing, missing/bad labels, spatial mismatch, training-only moments, transform alignment and archive/path attacks |
| `tests/test_models_config.py` | All model forward/backward integration, irregular dimensions, local skip and configuration validation |
| `tests/test_pipeline.py` | End-to-end training, safe reload, test evaluation, threshold search, timing, inference, plotting, SQLite and exact interrupted-resume equivalence |
| `.github/workflows/tests.yml` | CPU CI import/lint/unit/integration checks and JUnit export without restricted raw datasets |

## Configurations and evidence

`deepcrack.yaml` is the planned original-image research protocol; `crack500.yaml` requires missing original training data; `crack500-patches.yaml` explicitly selects the actual available patch protocol. `smoke.yaml` defines the small execution-only workload.

`datasets/manifests` supplies deterministic portable membership, `datasets/sources.json` source hashes and access/rights status, and `reports/*audit.json` actual integrity/claim audits. `reports/executed` includes genuine measurements and plot artifacts; it does not store invented benchmark reproductions. Original tables/images remain in `docs/manuscript` with provenance.

`pyproject.toml` and requirements files define package installation. README and architecture/evaluation/reproducibility documents provide the workflow and scientific boundaries. LICENSE and third-party notices explicitly separate new software from upstream assets.
