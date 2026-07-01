# Stage 14: 2048 Training Support

## Scope

This stage adds high-resolution training support for `2048x2048`.

New launch files:

- `run_train_csa_m2_topk64_2048.sh`
- `run_train_csa_all_layers_m2_topk64_2048_bs4.sh`

New config:

- `configs/config_all_layers_m2_topk64_2048_bs4.json`

## Code Changes

Training now supports query chunking:

- `--query-chunk-size`
- `--metrics-max-queries`

At `2048x2048`, image token count is much larger than at `512x512`. The training scripts compute the distillation loss by query chunks so the same entry can be used at high resolution.

For multi-layer training, each layer loss is backpropagated immediately with the correct scaling. This avoids keeping every layer's full computation graph alive before the optimizer step.

## Recommended 2048 Commands

Single-layer:

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64_2048.sh
```

All-layer:

```bash
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_bs4.sh
```

## Default 2048 Parameters

- `height=2048`
- `width=2048`
- `query_chunk_size=512`
- `metrics_max_queries=2048`
- all-layer `batch_size=4`

## Direct 2048 All-Layer Command

The default high-resolution all-layer launch is:

```bash
python experiments/z_image_indexer/train_csa_multilayer_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
  --prompt-file experiments/z_image_indexer/configs/default_prompts_train.txt \
  --output-dir experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_2048_bs4 \
  --steps 1000 \
  --height 2048 \
  --width 2048 \
  --num-inference-steps 4 \
  --layer-ids all \
  --batch-size 4 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --rank 128 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --recall-k 16 \
  --query-chunk-size 512 \
  --metrics-max-queries 2048 \
  --device cuda
```

## Multi-GPU Note

The current script is still single-process indexer-only training. Exposing multiple GPUs with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

does not by itself enable distributed data parallel or model parallel execution. The launch files are prepared for a multi-GPU server environment, but actual multi-process training support is a separate implementation step.
