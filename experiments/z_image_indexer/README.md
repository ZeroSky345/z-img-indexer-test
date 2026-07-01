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

## LoRA Adapter Weight Training

Train LoRA adapter weights while keeping the base Z-Image-Turbo DiT weights frozen:

```bash
bash experiments/z_image_indexer/run_train_lora_weights_2048_adapter.sh
```

This is the recommended entrypoint for preparing the LoRA weight used by the indexer teacher. It trains only LoRA adapter parameters and uses the public `ostris/zimage_turbo_training_adapter` preset to better preserve few-step Turbo behavior.

After training, select one checkpoint and use it as the teacher LoRA:

```bash
mkdir -p ./models/lora
cp ./models/lora/z_image_turbo_lora_2048_adapter/epoch-4.safetensors \
  ./models/lora/z_image_turbo_lora.safetensors
```

See `reports/STAGE15_LORA_WEIGHT_TRAINING.md`.

## LoRA-Teacher Indexer Training

Use the trained LoRA checkpoint as the teacher condition when distilling the indexer:

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64_2048_lora_teacher.sh
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh
```

By default these scripts use:

```bash
./models/lora/z_image_turbo_lora.safetensors
```

Override with `TEACHER_LORA_PATH=/path/to/lora.safetensors`.

See `reports/STAGE16_LORA_TEACHER_INDEXER.md`.

## 2048 training support

High-resolution entrypoints:

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64_2048.sh
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_bs4.sh
```

These use `--height 2048 --width 2048` plus `--query-chunk-size 512`. The all-layer entry uses `--batch-size 4`.

See `reports/STAGE14_2048_TRAINING_SUPPORT.md`.
