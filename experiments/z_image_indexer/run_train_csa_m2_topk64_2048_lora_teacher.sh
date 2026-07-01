#!/usr/bin/env bash
set -euo pipefail

# Train a single-layer CSA indexer using base Z-Image-Turbo + LoRA as teacher.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/tmp/modelscope-cache}
export HF_HOME=${HF_HOME:-/tmp/hf-home}

TEACHER_LORA_PATH=${TEACHER_LORA_PATH:-./models/lora/z_image_turbo_lora.safetensors}
TEACHER_LORA_ALPHA=${TEACHER_LORA_ALPHA:-1.0}

python experiments/z_image_indexer/train_csa_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
  --teacher-lora-path "${TEACHER_LORA_PATH}" \
  --teacher-lora-alpha "${TEACHER_LORA_ALPHA}" \
  --prompt-file experiments/z_image_indexer/configs/default_prompts_train.txt \
  --output-dir experiments/z_image_indexer/results/train_csa_layer12_m2_topk64_2048_lora_teacher \
  --steps 1000 \
  --height 2048 \
  --width 2048 \
  --num-inference-steps 4 \
  --layer-id 12 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --rank 128 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --recall-k 16 \
  --query-chunk-size 512 \
  --metrics-max-queries 2048 \
  --seed 42 \
  --device cuda
