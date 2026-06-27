#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export MODELSCOPE_CACHE=/tmp/modelscope-cache
export HF_HOME=/tmp/hf-home

python experiments/z_image_indexer/train_csa_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
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
