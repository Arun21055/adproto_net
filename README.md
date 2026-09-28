# ADProto-Net: Adaptive Dense Prototype Network

Few-shot abdominal organ segmentation in CT using **Multi-Scale Adaptive Prototype Aggregation (MSAPA)**.
Given **one labeled example** of an organ, the model segments that organ in new patients' CT scans.

> Developed during an AI/ML research internship at NIT Tiruchirappalli (May–June 2026).

## Idea

Standard prototypical networks compute a single class prototype by masked average pooling at **one spatial scale**.
MSAPA instead:

1. Pools support features at **three scales** (32×32, 16×16, 8×8) and extracts a foreground and a background prototype at each scale.
2. Uses a lightweight **channel-attention gate**, conditioned on the *query* image, to produce soft weights over the three scales.
3. Combines the three prototypes with those weights, so each query image decides which scale to trust.

This adds only ~130K parameters and avoids expensive HW×HW self-attention.

## Architecture

```
Support image + mask ─┐
                      ├─▶ ResNet50 (stem + layer1–3) ─▶ ASPP ─▶ Decoder ─▶ features
Query image ──────────┘

Support features + mask ─▶ MSAPA (3 scales + query-conditioned gate) ─▶ fg / bg prototype
Query features ─▶ cosine similarity to fg / bg prototype ─▶ SegHead ─▶ mask logits
                                        └─▶ Auxiliary head (deep supervision)
```

- **Backbone:** ImageNet-pretrained ResNet50 → ASPP → Decoder with a skip connection
- **Loss:** focal-Tversky + BCE (with an auxiliary loss, weight 0.4)
- **Optimiser:** AdamW, lr 2e-4, cosine schedule with warm-up, mixed precision (AMP)

## Results

1-shot, 4-fold patient-level cross-validation, volume-level Dice (%):

| Method | Spleen | R. Kidney | L. Kidney | Liver | Aorta | Mean |
|---|---|---|---|---|---|---|
| Baseline (single-scale, no MSAPA) | 88.96 | 79.77 | 83.51 | 87.56 | 84.56 | 84.87 |
| **ADProto-Net (MSAPA)** | **91.70** | **85.47** | **88.71** | 84.80 | **85.03** | **87.14** |

MSAPA improves the mean Dice by **2.27 points**, with the largest gains on the kidneys (about +5 points).
The liver drops slightly (−2.76); a likely cause is that finer scales dilute the global context that large organs need. Analysing the learned gate weights per organ is future work.

## Evaluation protocol

- **Patient-level 4-fold cross-validation.** No patient appears in both training and validation. Patient IDs are shuffled with a fixed seed (`--split_seed`, default 42) before splitting, so hard cases are spread across folds.
- **Support and query come from different patients.** Validation support examples are fixed with a seed (`--eval_seed`, default 2024) so results are reproducible.
- **Metric:** volume-level Dice per organ, averaged over patients, then over folds.
- **Post-processing:** hole filling and largest connected component per predicted volume.
- Organs: spleen, right kidney, left kidney, liver, aorta (label IDs 1, 2, 3, 6, 8).

## Data

Multi-organ abdominal CT volumes in NIfTI format (`.nii` / `.nii.gz`) with matching label volumes.
`--mode preprocess` converts each patient to a compressed `.npz` file (`img`, `label`, `pid`, `n_slices`).
Slices are windowed to [−125, 275] HU, normalised to [0, 1] and resized to 256×256.

> Add the dataset name and citation here.

## Setup

```bash
pip install -r requirements.txt
```

Set data locations with environment variables (or the command-line flags below):

```bash
export ADPROTO_RAW_DIR=/path/to/RawData
export ADPROTO_DATA_DIR=/path/to/Preprocessed
export ADPROTO_SAVE_DIR=/path/to/checkpoints
```

## Usage

```bash
# 1. Preprocess raw NIfTI volumes
python adproto_net.py --mode preprocess

# 2. Train with 1-shot, all 4 folds
python adproto_net.py --mode train --fold -1 --n_shot 1

# Try a different patient split
python adproto_net.py --mode train --fold -1 --n_shot 1 --split_seed 7

# 3. Evaluate saved checkpoints
python adproto_net.py --mode eval --fold -1
```

Useful flags: `--n_iter`, `--lr`, `--batch`, `--tta`, `--msapa_scales 1 2 4`, `--raw_dir`, `--data_dir`, `--save_dir`.

**Note:** if you change `--split_seed`, delete the `.ram_cache_adproto/` folder inside your data directory first. Cached fold contents are keyed by fold number only and do not update automatically.

## Limitations and future work

- Evaluated on 5 organs and one dataset; no external validation.
- Slice-based 2D model; a 3D or 2.5D variant may use context better.
- Liver performance decreases slightly compared with the baseline.
- Future work: per-organ scale analysis of the gate, more shots (n > 1), other modalities such as MRI.

## Repository structure

```
adproto_net.py     # model, data pipeline, training and evaluation
requirements.txt
README.md
```

## Author

**P R Arun Kumar**, VIT-AP University
