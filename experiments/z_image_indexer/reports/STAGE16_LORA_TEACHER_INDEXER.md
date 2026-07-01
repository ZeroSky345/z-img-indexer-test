# Stage 16: LoRA-Teacher Indexer Training

## Scope

This stage adds support for training CSA/indexer modules against a LoRA-adapted Z-Image-Turbo teacher.

New training parameters:

- `--teacher-lora-path`
- `--teacher-lora-alpha`

New scripts:

- `run_train_csa_m2_topk64_2048_lora_teacher.sh`
- `run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh`
- `run_eval_csa_m2_topk64_2048_lora_teacher.sh`
- `run_eval_csa_all_layers_m2_topk64_2048_lora_teacher.sh`

New eval entry:

- `eval_csa_multilayer_indexer.py`

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

1. Prepare an external LoRA weight:

```bash
mkdir -p ./models/lora
cp /path/to/z_image_turbo_lora.safetensors ./models/lora/z_image_turbo_lora.safetensors
```

2. Train a single-layer LoRA-teacher indexer:

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64_2048_lora_teacher.sh
```

3. Evaluate the single-layer LoRA-teacher indexer:

```bash
bash experiments/z_image_indexer/run_eval_csa_m2_topk64_2048_lora_teacher.sh
```

4. Train the all-layer LoRA-teacher indexer:

```bash
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh
```

5. Evaluate the all-layer LoRA-teacher indexer:

```bash
bash experiments/z_image_indexer/run_eval_csa_all_layers_m2_topk64_2048_lora_teacher.sh
```

## Default LoRA Checkpoint

The indexer scripts default to:

```bash
./models/lora/z_image_turbo_lora.safetensors
```

Override it with:

```bash
TEACHER_LORA_PATH=/path/to/lora.safetensors \
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh
```

## What Is Trained

The LoRA is loaded into the frozen teacher DiT. The only trainable module remains the indexer.

This project does not train the Z-Image-Turbo LoRA. The LoRA is treated as an external teacher weight.

This is not LoRA-on-indexer. The indexer itself is already small, so training its full parameters is simpler and cleaner than adding LoRA to it.

## Full-Layer Evaluation

`eval_csa_multilayer_indexer.py` loads the multi-layer indexer checkpoint, applies sparse replacement to all requested layers, and compares dense vs sparse generation under the same LoRA-loaded teacher model.

Default all-layer eval checkpoint:

```bash
experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4/csa_multilayer_indexer_distill.pt
```

Override it with:

```bash
INDEXER_CKPT=/path/to/csa_multilayer_indexer_distill.pt \
bash experiments/z_image_indexer/run_eval_csa_all_layers_m2_topk64_2048_lora_teacher.sh
```
