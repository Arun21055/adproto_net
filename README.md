# ADProto-Net — Adaptive Dense Prototype Network

Few-shot segmentation model with **Multi-Scale Adaptive Prototype Aggregation (MSAPA)**.

Standard prototypical matching uses a single masked-average-pool at one spatial
scale. MSAPA instead pools at three pyramid scales (32x32, 16x16, 8x8),
producing three prototype candidates per class, and combines them with a
lightweight channel-attention gate that adaptively weights each scale's
contribution per query image.

## Architecture
- ResNet50 (stem + layer1–3) → ASPP → Decoder
- Support branch: MSAPA → `fg_proto` + `bg_proto` (B, 256)
- Query branch: cosine similarity maps → SegHead → upsampled logits
- Auxiliary head on coarse features for deep supervision
- Loss: focal-Tversky

## Usage
```bash
python adproto_net.py --mode train --fold -1 --n_shot 1
python adproto_net.py --mode train --fold -1 --n_shot 1 --split_seed 7
python adproto_net.py --mode eval  --fold -1
```

## Notes
- `split_folds()` shuffles patient IDs (fixed, configurable seed) before
  slicing into folds, rather than using sorted contiguous blocks.
- `cfg.split_seed` (default 42) controls the fold shuffle.
- If you change `split_seed`, delete the old `.ram_cache_adproto/` directory
  under `data_dir` before re-running — cached fold contents are keyed by
  fold number only and won't auto-invalidate on a split change.
