# Stage 16: LoRA-Teacher Indexer Training

## Scope

This stage adds support for training CSA/indexer modules against a LoRA-adapted Z-Image-Turbo teacher.

New training parameters:

- `--teacher-lora-path`
- `--teacher-lora-alpha`

New scripts:

- `run_train_csa_m2_topk64_2048_lora_teacher.sh`
- `run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh`

New config:

- `configs/config_all_layers_m2_topk64_2048_lora_teacher_bs4.json`

## Why This Is Needed

If final inference uses:

```text
Z-Image-Turbo + LoRA + sparse/indexer replacement
```

then the indexer should be distilled from the same teacher:

```text
Z-Image-Turbo + LoRA
```

Training the indexer against the base model and then applying LoRA later can create a mismatch between the teacher attention distribution and the actual inference distribution.

## Recommended Sequence

1. Train the LoRA:

```bash
bash experiments/z_image_indexer/run_train_z_image_turbo_lora_2048_adapter.sh
```

2. Train a single-layer LoRA-teacher indexer:

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64_2048_lora_teacher.sh
```

3. Train the all-layer LoRA-teacher indexer:

```bash
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh
```

## Default LoRA Checkpoint

The indexer scripts default to:

```bash
./models/train/Z-Image-Turbo_lora_2048_adapter/epoch-4.safetensors
```

Override it with:

```bash
TEACHER_LORA_PATH=/path/to/lora.safetensors \
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh
```

## What Is Trained

The LoRA is loaded into the frozen teacher DiT. The only trainable module remains the indexer.

This is not LoRA-on-indexer. The indexer itself is already small, so training its full parameters is simpler and cleaner than adding LoRA to it.
