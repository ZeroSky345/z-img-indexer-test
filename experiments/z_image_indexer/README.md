# Z-Image Indexer Experiments

## Structure

- `csa_common.py`
  - shared helpers for CSA-style training/evaluation
- `train_*.py`, `eval_*.py`, `stage*.py`
  - executable experiment and evaluation scripts
- `run_*.sh`
  - direct shell entrypoints for server-side execution
- `configs/`
  - recommended config snapshots and prompt files
- `reports/`
  - human-readable experiment conclusions and summaries
- `results/`
  - JSON metrics and summaries grouped by stage
- `comparisons/`
  - side-by-side dense vs sparse image comparisons grouped by stage

## Current default training candidate

- model: `Z-Image-Turbo`
- layer: `12`
- route: `CSA-like`
- `compression_rate=2`
- `compressed_topk=64`
- resolution: `2048x2048`

See:

- `reports/TRAINING_QUICKSTART.md`
- `configs/config_recommended_m2_topk64.json`

## All-layer training support

`train_csa_multilayer_indexer.py` supports all-layer indexer-only training:

```bash
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_bs4.sh
```

Important options:

- `--layer-ids all` trains one indexer per main Z-Image DiT layer
- `--batch-size 4` accumulates four prompt/timestep/latent samples per optimizer step
- `--height 2048 --width 2048` is the current high-resolution target
- `--layer-ids 0-29` and mixed ranges such as `0,4,8-13` are also supported

See `reports/STAGE14_2048_TRAINING_SUPPORT.md`.

## Z-Image-Turbo LoRA 2048

LoRA entrypoints:

```bash
bash experiments/z_image_indexer/run_train_z_image_turbo_lora_2048_adapter.sh
bash experiments/z_image_indexer/run_train_z_image_turbo_lora_2048.sh
```

The adapter-assisted path is recommended first for Turbo.

See `reports/STAGE15_Z_IMAGE_TURBO_LORA_2048.md`.

## 2048 training support

High-resolution entrypoints:

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64_2048.sh
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_bs4.sh
```

These use `--height 2048 --width 2048` plus `--query-chunk-size 512`. The all-layer entry uses `--batch-size 4`.

See `reports/STAGE14_2048_TRAINING_SUPPORT.md`.
