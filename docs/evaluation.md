# Evaluation conventions and statistical limits

## Counts and aggregation

Crack is the positive class. Prediction uses `sigmoid(logit) >= threshold`; ground truth uses mask>0. Each evaluation accumulates TP,FP,FN,TN across all evaluated pixels. Precision, recall, F1, crack IoU, background IoU, two-class mIoU, pixel accuracy and foreground prevalence are derived from those counts. Output counts allow independent verification.

Undefined ratios are JSON null, not silently zero or perfect scores. If a class has zero IoU denominator, mIoU averages the defined classes. Empty foreground with a correct all-background prediction therefore has null foreground F1 and mIoU1; report prevalence with these scores. With foreground present, all-background prediction has F1 zero despite potentially high accuracy. Unit tests exercise these cases.

The data are resized to the configured experimental resolution for ordinary evaluation. Output cropping excludes padding. Native-resolution tiled inference is a different mode, not mixed into the benchmark score denominator. No tolerance-dilation metric, label thickening or hidden morphological postprocessing is used. Edge benchmarks such as ODS/OIS with distance tolerance are not directly interchangeable with these exact pixel metrics.

## Selection versus testing

Checkpoint choice is maximum validation F1. Loss weights and the main threshold0.5 are fixed before test evaluation. Optional threshold search evaluates only validation. Learning-rate tuning selects on validation and never automatically evaluates test. Repeated tuning and checkpoint selection consume validation information; predefine search grids and reserve the test set for the final frozen protocol.

Predefined model/seed and ablation suites apply common data and training settings. Different model capacities and compute budgets still require disclosure; identical epochs do not guarantee equal optimization budgets. The explicit baseline implementations are useful comparisons but are not a reconstruction of undocumented manuscript code.

## Uncertainty

Evaluation stores per-image confusion records and identities. Bootstrap resamples whole images or inferred source groups, recalculating aggregate metrics for each draw. The default interval is percentile95 with1000 iterations. Smoke verification uses200 for execution economy, recorded in each report. Inferred groups are not certified scenes and may group unrelated names or miss related acquisitions.

Paired comparisons require matching sample IDs and identical image/mask checksums. Their image bootstrap quantifies sample variation conditional on two trained models. It does not quantify training instability or eliminate scene correlation. Full scientific comparisons should combine independent seeds with an appropriate hierarchical sampling/modeling strategy. No significance claim is drawn from the executed three-epoch runs.

## Performance measurements

The benchmark command records model-only batch1 timing, warmup/repetitions, median/p95 latency, mean-throughput FPS, parameter count, input resolution and environment. CUDA timing synchronizes the device. CPU results differ from the manuscript's RTX2080Ti claims and exclude loading, normalization and visualization. FLOPs and field-throughput are not estimated or falsely labeled as measured.

## Conditional manuscript audit

`scripts/audit_manuscript_metrics.py` checks F1 from reported precision/recall and derives mIoU/prevalence under a global confusion assumption. Some rows imply a large crack prevalence inconsistent with a typical sparse-crack task, or differ in implied mIoU. Such differences could reflect undocumented averaging or a numerical problem; raw logs are required to resolve them. The code preserves original claims rather than replacing them with invented observations.
