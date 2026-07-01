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

1. Train or prepare a LoRA adapter-weight checkpoint.

Recommended LoRA weight training command:

```bash
bash experiments/z_image_indexer/run_train_lora_weights_2048_adapter.sh
```

This trains only LoRA adapter weights. The base Z-Image-Turbo DiT weights stay frozen.

2. Put the selected LoRA checkpoint at the default teacher path:

```bash
mkdir -p ./models/lora
cp ./models/lora/z_image_turbo_lora_2048_adapter/epoch-4.safetensors \
  ./models/lora/z_image_turbo_lora.safetensors
```

3. Train a single-layer LoRA-teacher indexer:

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64_2048_lora_teacher.sh
```

4. Evaluate the single-layer LoRA-teacher indexer:

```bash
bash experiments/z_image_indexer/run_eval_csa_m2_topk64_2048_lora_teacher.sh
```

5. Train the all-layer LoRA-teacher indexer:

```bash
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh
```

6. Evaluate the all-layer LoRA-teacher indexer:

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

There are two separate trainable targets:

- LoRA adapter-weight training: trains only LoRA adapter parameters attached to the frozen Z-Image-Turbo DiT.
- Indexer distillation: loads the selected LoRA into the frozen teacher DiT and trains only the CSA/indexer parameters.

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
