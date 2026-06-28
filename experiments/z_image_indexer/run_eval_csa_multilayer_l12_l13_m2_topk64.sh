#!/usr/bin/env bash
set -euo pipefail

# Evaluation template for the first controlled multi-layer validation run.

export CUDA_VISIBLE_DEVICES=0
export MODELSCOPE_CACHE=/tmp/modelscope-cache
export HF_HOME=/tmp/hf-home

python experiments/z_image_indexer/eval_csa_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
  --prompt-file experiments/z_image_indexer/configs/benchmark_prompts_v1.txt \
  --indexer-ckpt experiments/z_image_indexer/results/train_csa_layers12_13_m2_topk64/csa_indexer_distill.pt \
  --output-dir experiments/z_image_indexer/results/eval_csa_layers12_13_m2_topk64 \
  --height 512 \
  --width 512 \
  --num-inference-steps 4 \
  --layer-id 12 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --seed 42 \
  --device cuda
