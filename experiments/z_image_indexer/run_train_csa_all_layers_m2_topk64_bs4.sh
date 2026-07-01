#!/usr/bin/env bash
set -euo pipefail

# All-layer CSA-style indexer distillation.
# This is indexer-only training. Start on a single 80GB GPU for a short smoke run,
# then move to multi-GPU or reduce batch size if memory pressure is high.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/tmp/modelscope-cache}
export HF_HOME=${HF_HOME:-/tmp/hf-home}

python experiments/z_image_indexer/train_csa_multilayer_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
  --prompt-file experiments/z_image_indexer/configs/default_prompts_train.txt \
  --output-dir experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_bs4 \
  --steps 1000 \
  --height 512 \
  --width 512 \
  --num-inference-steps 4 \
  --layer-ids all \
  --batch-size 4 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --rank 128 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --recall-k 16 \
  --log-every 10 \
  --seed 42 \
  --device cuda
