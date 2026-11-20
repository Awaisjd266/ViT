# Dataset card

## Intended task and sources

Pixel-level foreground crack segmentation on pavement/concrete surface RGB photographs. These are existing datasets, not a dataset created by the manuscript authors or by this software.

- **DeepCrack (Liu et al.2019)**:537 image–mask pairs, published official300 training and237 test. Source: https://github.com/yhlleo/DeepCrack. Dataset archive has `train_img/train_lab/test_img/test_lab`. The prepared training/validation membership is240/60, selected with seed42 while preserving inferred source scenes and exact decoded duplicate groups. Actual image dimensions are audited;544×384 is the manuscript's description, not an unverified assertion about every file.
- **Crack500 (Yang et al.)**: source https://github.com/fyangneil/pavement-crack-detection. The author's official1.98GB bundle contains1896 train,348 validation and1124 test image crops, plus50 original validation and200 original test images. It does **not** contain an original training-image archive. The500-original250/50/200 protocol remains incomplete.

## Label and preprocessing conventions

Masks are black background and white crack, with permitted raw values0/1/255. Values>0 become crack only after the audit confirms binary source encoding. Mask dimensions must match the corresponding image. Nearest-neighbor mask resizing preserves class values. RGB uses bilinear resizing and training-only channel normalization. Augmented pairs remain within their source split; no augmentation is counted as a new independent observation.

## Integrity and independence

Store raw-file SHA256 and decoded RGB hashes. Corrupt or missing pairs, differing geometry, invalid labels and exact decoded cross-split duplicates fail validation. The acquired releases passed exact duplicate checks. Source-scene prefixes identified three possible DeepCrack cross-split groups and six Crack500 crop cross-split groups. These are naming-based flags, not proof of either independence or identity. Official assignments are preserved and warnings retained. Full source-disjoint secondary analysis and more powerful near-duplicate/scene verification remain recommended before strong generalization claims.

## Licensing and restrictions

DeepCrack's source README explicitly restricts use to noncommercial research and educational purposes and states partial original-image copyright ownership. Crack500's source README requests citations and links public archives but supplies no blanket redistribution license. The MIT license in this repository covers new software only; it does not relicense dataset content or original manuscript figures. No complete raw dataset is published through ordinary Git by this implementation without established redistribution rights.

## Scope limitations

Neither dataset supplies the four image-level categories requested initially; no such labels are manufactured. Longitudinal/transverse/alligator categories would require validated additional annotation and quantities not given in AS.docx. Source-domain overlap, shared acquisition scenes, variable crack widths, shadows, mask thickness and resizing affect scores. The benchmark is not a representative field safety evaluation and contains no measured crack width in physical units. Image-level accuracy claims, class-balanced evaluation and tolerant edge metrics must not be mixed with exact dense pixel metrics.
