# Stage 15: LoRA Adapter Weight Training

## Scope

This stage adds the training entrypoint for LoRA adapter weights used by the indexer pipeline.

The intent is:

```text
frozen Z-Image-Turbo base DiT + trainable LoRA adapter weights
```

It is not full-parameter Z-Image-Turbo training. `--lora_base_model dit` means the LoRA modules are attached to the DiT, while the base DiT parameters remain frozen and only LoRA adapter parameters are optimized.

## Scripts

- `run_train_lora_weights_2048_adapter.sh`
- `run_train_lora_weights_2048.sh`

The adapter version is recommended because Z-Image-Turbo is a distilled acceleration model. The script loads `ostris/zimage_turbo_training_adapter` as `--preset_lora_path` to reduce the risk of damaging few-step behavior during LoRA adapter-weight training.

## Recommended Command

```bash
bash experiments/z_image_indexer/run_train_lora_weights_2048_adapter.sh
```

Default output:

```text
./models/lora/z_image_turbo_lora_2048_adapter/
```

After training, choose a checkpoint and use it as the teacher LoRA for indexer distillation:

```bash
mkdir -p ./models/lora
cp ./models/lora/z_image_turbo_lora_2048_adapter/epoch-4.safetensors \
  ./models/lora/z_image_turbo_lora.safetensors
```

or:

```bash
export TEACHER_LORA_PATH=./models/lora/z_image_turbo_lora_2048_adapter/epoch-4.safetensors
```

## Downstream Flow

```text
1. Train LoRA adapter weights.
2. Select one LoRA checkpoint.
3. Load that LoRA into the frozen teacher DiT.
4. Distill CSA/indexer attention routing from the LoRA-adapted teacher.
5. Evaluate dense vs sparse generation under the same LoRA condition.
```

## Notes

- The LoRA output is a `.safetensors` adapter checkpoint, not a merged full model.
- Indexer training still optimizes the indexer itself, not LoRA-on-indexer.
- For 2048 training, keep `--height 2048 --width 2048 --max_pixels 4194304`.
