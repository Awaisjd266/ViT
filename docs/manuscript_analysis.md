# Manuscript analysis and implementation decisions

Source: **AS.docx**, “Enhancing Road Pavement Crack Detection Using a Vision Transformer Approach”, Muhammad Awais Javeed, Zijie Zhou and Dawei Li. SHA-256: `99eccf2a6e8a70f92da8540d8d4d088d6a30d67baea64b2db868371b15cb4fee`.

The full document body, every table, the native Word mathematical objects and all four embedded figures were inspected before implementation. The source contains 318 paragraphs including table cells, 20 tables (16 mathematical layout tables and four data tables), and 49 mathematical objects including inline expressions. This is an independent implementation; the manuscript provides no source code, weights, training logs or exact sample manifest. Its observations cannot be independently authenticated from the document.

## Section by section

| Source section | Research content | Engineering decision |
|---|---|---|
| Abstract and Introduction | Binary road-crack segmentation; global context motivation; accuracy claims 90.05/91.78 percent | Preserve dense prediction objective; require foreground metrics rather than accuracy alone |
| Related work 2.1 | Edges, thresholding and morphology | Optional classical threshold benchmark; no claim it reproduces an unspecified manuscript baseline |
| Related work 2.2 | CNN, residual networks and U-Net | Define explicit segmentation baselines; generic architecture names are insufficient for exact reproduction |
| Related work 2.3 | Global ViT and shifted-window context | Implement global patch attention; do not claim Swin implementation |
| 3 and Figure 2 | RGB + mask, ViT encoder and progressively upsampled decoder | Modular PyTorch encoder/decoder; learned interpolated positional grid; no class token |
| 3.1 and equations 1–2 | Training-only mean/std; paired rotations and flips | Train-only streaming channel moments; synchronized geometry; nearest mask resampling; original split retained |
| 3.2 and equations 3–6 | Patch embeddings, pre-layer-normalized attention and MLP residuals | Native TransformerEncoderLayer with norm_first=True, GELU and scaled dot-product attention; square-root scaling verified in Word XML |
| 3.3 and equations 7–10 | Token reshape, staged upsampling, sigmoid and threshold | Pad to patch multiple, interpolate position grid, bilinear upsampling + convolutions, crop logits; default threshold 0.5 |
| 3.4 and equations 11–13 | Equal BCE and Dice; prediction/target overlap | Stable BCEWithLogits + per-image soft Dice with true targets; native Word accents distinguish prediction from target (plain text extraction loses them). No unsupported claim that the printed Dice formula is mathematically wrong |
| 4.1 | RTX 2080 Ti, Ubuntu20.04, Python3.6, torch1.10 | Historical environment claim retained in provenance; executable implementation uses supported Python>=3.10 and pinned torch2.6; actual host captured per run |
| 4.2 and Table 1 | Crack500 250/50/200; DeepCrack official300/237 with training240/val60 | Correct DeepCrack test count 37→237; no four semantic categories exist; user confirmed these two datasets and counts |
| 4.3 and equations 14–20 | Binary confusion counts, P/R/F1, accuracy, two-class mIoU | Exact int64 counts; explicit undefined policy; crack IoU and background IoU separate; dataset aggregate and image-level reporting |
| 4.4 and Tables 2–3 | Five methods, per-dataset metrics and FPS | Preserve as unverified manuscript claims, machine-readable CSV; no target-driven tuning or fabricated reproduction |
| 4.5–4.6 and Table 4 | Dataset-dependent advantages, no multi-run uncertainty, speed claims | Validate arithmetic; paired image bootstrap, seed suites, controlled baselines and timing with warmup/synchronization |
| Conclusion | Thin cracks, deployment and domain shifts | Optional local detail skip, arbitrary-size tiled inference, separate cross-domain evaluation; these are additions |
| Contributions, declarations and acknowledgments | Authorship and grant attribution | Retain manuscript attribution without implying authors endorse this implementation or funding supported the software |
| Data availability and references | Public benchmark sources, data on request | Source/licensing manifest, checksums and acquisition logs; public availability is not a blanket redistribution license |

## Figures examined

1. Example grayscale-looking pavement RGB image and a thin white foreground mask. No recoverable raw sample ID is provided.
2. A five-stage workflow diagram, not four dataset categories. Figure lists optimizer alternatives and unspecified training settings, and draws Add/Norm after attention; equations specify pre-norm. Equations control this implementation. Diagram permits unified data, while Discussion explicitly says each model was trained separately per dataset: per-dataset training is the main protocol; pooled training is a labeled additional experiment.
3. Five Crack500 examples, masks and six method rows, with the proposed method rendered magenta. This is a qualitative manuscript illustration, not a stored prediction tensor or a new experiment.
4. Five DeepCrack examples with the same presentation; binary colors differ across rows. Neither montage is used as training data, ground truth or proof of numerical metrics.

## Mathematical and experimental cautions

- The symbols P and N are overloaded between patch width/probability and token/pixel counts. Code uses explicit names and tensor dimensions.
- The paper's 640×360 Crack500 dimensions are width×height. Height360 is not divisible by patch16. Padding/cropping is explicit, so boundary pixels are retained and metrics exclude padding.
- Micro F1 must equal 2PR/(P+R); foreground IoU equals F1/(2−F1). Two-class mIoU also requires background IoU. A table using per-image averaged metrics need not obey global identities; aggregation is unspecified in AS.docx. An audit reports conditional inconsistencies, not invented corrections to experimental measurements.
- Pixel accuracy may be high on all-background outputs. Report foreground prevalence and foreground metrics; empty foreground scores use explicit conventions.
- Images with common source-scene IDs may be correlated even when byte-distinct. Audit exact and decoded hashes plus inferred source groups. Official splits are not silently reshuffled to force clean counts; conflicts cause a visible audit failure requiring a documented protocol choice.
- A single run establishes neither statistical superiority nor cross-domain generalization. Separate-dataset training cannot test domain transfer. A short smoke run cannot reproduce published metrics.

## Missing settings and selected defaults

Patch16, embedding192, six layers, six heads, MLP ratio4, dropout0.1, decoder128; AdamW lr3e-4/decay0.01, cosine schedule, batch4, 100 epochs, early stopping20, clip1, three comparison seeds. All are engineering selections, **not original experimental settings**. Baseline definitions are explicit in architecture documentation. No pretrained weights are downloaded by default. Optional class weighting is train-only and disabled for the main equal-weight loss protocol.

## Scope and evidence

Complete code execution, dataset acquisition status, smoke experiments and tests are recorded separately in `reports/verification.json` and `reports/`. No benchmark reproduction is asserted unless the full protocol has actually run. There is no reason to add a service database for 1037 image pairs: portable manifests plus an optional SQLite experiment registry cover data identity and run tracking without operating an unnecessary database server.
