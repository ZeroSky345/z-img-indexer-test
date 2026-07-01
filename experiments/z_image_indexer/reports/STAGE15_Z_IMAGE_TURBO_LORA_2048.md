# Stage 15: Z-Image-Turbo LoRA 2048 Entry

## Scope

This stage adds direct LoRA training entrypoints for `Z-Image-Turbo` at `2048x2048`.

New scripts:

- `run_train_z_image_turbo_lora_2048.sh`
- `run_train_z_image_turbo_lora_2048_adapter.sh`

New config:

- `configs/config_z_image_turbo_lora_2048.json`

## What This Trains

This is different from the CSA/indexer experiments.

- LoRA training updates low-rank adapter weights on the Z-Image DiT.
- CSA/indexer training learns an external sparse-routing indexer.
- LoRA changes generation behavior or style adaptation.
- The indexer changes which attention memory blocks are selected during sparse replacement.

## Recommended Command

For `Z-Image-Turbo`, use the adapter-assisted path first:

```bash
bash experiments/z_image_indexer/run_train_z_image_turbo_lora_2048_adapter.sh
```

The plain LoRA path is also available:

```bash
bash experiments/z_image_indexer/run_train_z_image_turbo_lora_2048.sh
```

## Key Parameters

- `height=2048`
- `width=2048`
- `max_pixels=4194304`
- `lora_base_model=dit`
- `lora_target_modules=to_q,to_k,to_v,to_out.0,w1,w2,w3`
- `lora_rank=32`
- `learning_rate=1e-4`
- `num_epochs=5`
- `NUM_PROCESSES=4` by default

## Why Adapter-Assisted LoRA Exists

`Z-Image-Turbo` is a distilled few-step model. Ordinary supervised LoRA training can weaken the model's distilled acceleration behavior, especially under 4-step inference.

The adapter-assisted script loads:

```bash
models/ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v1.safetensors
```

as a preset LoRA before training. This follows the upstream example's recommendation for Turbo training.

## Practical Sequence With Indexer Training

Recommended experiment order:

1. Train LoRA at 2048.
2. Validate LoRA generation quality with the standard Z-Image LoRA validation script.
3. Freeze the base model plus LoRA behavior.
4. Train the CSA/indexer against that fixed teacher if the final target is sparse inference with the LoRA behavior preserved.

This avoids training the indexer against a teacher attention distribution that later changes after LoRA is added.
