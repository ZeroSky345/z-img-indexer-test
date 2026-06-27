# CSA Indexer Training Quickstart

## Goal

This directory now contains a first formalized training/evaluation path for the current recommended candidate:

- model: `Z-Image-Turbo`
- layer: `12`
- route: `CSA-like`
- `compression_rate=2`
- `compressed_topk=64`

This is still **indexer-only training**. It does not yet retrain the main `Z-Image` weights.

## Files

- `train_csa_indexer.py`
  - formalized training entry for CSA-style indexer distillation
- `eval_csa_indexer.py`
  - post-training dense-vs-CSA evaluation entry
- `csa_common.py`
  - shared helpers
- `config_recommended_m2_topk64.json`
  - recommended default configuration reference
- `default_prompts_train.txt`
  - default prompt set

## Environment assumptions

These commands assume:

- the repo is already copied to the training server
- the server already has a Python environment with `diffsynth` dependencies available
- model weights are accessible under the provided model base path

Recommended environment variables:

```bash
export CUDA_VISIBLE_DEVICES=0
export MODELSCOPE_CACHE=/tmp/modelscope-cache
export HF_HOME=/tmp/hf-home
```

## Recommended training command

```bash
python experiments/z_image_indexer/train_csa_indexer.py \
  --model-base-path /path/to/models \
  --prompt-file experiments/z_image_indexer/default_prompts_train.txt \
  --output-dir experiments/z_image_indexer/results/train_csa_layer12_m2_topk64 \
  --steps 1000 \
  --height 512 \
  --width 512 \
  --num-inference-steps 4 \
  --layer-id 12 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --rank 128 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --recall-k 16 \
  --seed 42 \
  --device cuda
```

Main outputs:

- `csa_indexer_distill.pt`
- `metrics.json`
- `summary.json`
- `run_config.json`

## Recommended evaluation command

Run this after training finishes:

```bash
python experiments/z_image_indexer/eval_csa_indexer.py \
  --model-base-path /path/to/models \
  --prompt-file experiments/z_image_indexer/default_prompts_train.txt \
  --indexer-ckpt experiments/z_image_indexer/results/train_csa_layer12_m2_topk64/csa_indexer_distill.pt \
  --output-dir experiments/z_image_indexer/results/eval_csa_layer12_m2_topk64 \
  --height 512 \
  --width 512 \
  --num-inference-steps 4 \
  --layer-id 12 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --seed 42 \
  --device cuda
```

Main outputs:

- `records.json`
- `summary.json`
- `run_config.json`
- dense / sparse output images
- side-by-side comparison images

## Suggested workflow on a training server

1. Prepare model cache path and environment variables.
2. Run `train_csa_indexer.py`.
3. Inspect `summary.json` and `metrics.json`.
4. Run `eval_csa_indexer.py`.
5. Compare the new `summary.json` against the current pre-training baseline:
   - latency
   - sparse/dense ratio
   - image MSE

## What this does not yet do

- multi-layer training
- joint training with the base model
- LPIPS / OCR / larger benchmark evaluation

Those are next steps after this first formalized training entry is stable.
