# Dataset acquisition and organization

This dedicated directory integrates authentic benchmark acquisition with the research code. It does not create new crack photographs or human annotations.

```text
datasets/
├── sources.json
├── DATASET_CARD.md
├── manifests/
│   ├── deepcrack.csv
│   └── crack500-patches.csv
├── downloads/                    # Local archives, ignored
│   ├── DeepCrack.zip
│   └── pavement-crack-datasets.zip
└── raw/                          # Local image/mask pairs, ignored
    ├── deepcrack/
    │   ├── train_img/            # 300 original images
    │   ├── train_lab/            # 300 original masks
    │   ├── test_img/             # 237 original images
    │   └── test_lab/             # 237 original masks
    └── crack500/
        ├── traincrop/           # 1896 image/mask pairs
        ├── valcrop/             # 348 pairs
        ├── testcrop/            # 1124 pairs
        ├── valdata/             # 50 original images and masks
        ├── testdata/            # 200 original images and masks
        └── source_inventory.json
```

Read `README.md` at repository root for exact acquisition/preparation commands. Paths in manifests are relative to the selected dataset root. Statistics, augmentation and models consume the manifest rather than relying on folder enumeration order.

The four DeepCrack directories are file roles, not four crack categories. Both datasets are binary pixel segmentation. The user confirmed the intended dataset names/splits. DeepCrack's test count is237, correcting the manuscript table typo37.

The available Crack500 patch release is scientifically distinct from500 full originals resized to640×360. The loader enforces1896/348/1124 for that alternative and250/50/200 for original-image mode. It fails on missing counts rather than padding with copies, generated images or unverified labels.

Public download access does not establish permission to republish complete archives. Raw data are locally present where the acquisition report says so and are excluded by `.gitignore`. Keep upstream notices. Obtain explicit applicable redistribution rights before mirroring raw data in a public GitHub repository; a manifest/download workflow is provided meanwhile.
