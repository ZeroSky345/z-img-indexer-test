#!/usr/bin/env bash
set -euo pipefail

# Evaluate all-layer CSA indexer replacement using the selected/trained LoRA weight.
# for both dense and sparse generation.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/tmp/modelscope-cache}
export HF_HOME=${HF_HOME:-/tmp/hf-home}

TEACHER_LORA_PATH=${TEACHER_LORA_PATH:-./models/lora/z_image_turbo_lora.safetensors}
TEACHER_LORA_ALPHA=${TEACHER_LORA_ALPHA:-1.0}
INDEXER_CKPT=${INDEXER_CKPT:-experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4/csa_multilayer_indexer_distill.pt}

python experiments/z_image_indexer/eval_csa_multilayer_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
  --teacher-lora-path "${TEACHER_LORA_PATH}" \
  --teacher-lora-alpha "${TEACHER_LORA_ALPHA}" \
  --prompt-file experiments/z_image_indexer/configs/benchmark_prompts_v1.txt \
  --indexer-ckpt "${INDEXER_CKPT}" \
  --output-dir experiments/z_image_indexer/results/eval_csa_all_layers_m2_topk64_2048_lora_teacher \
  --height 2048 \
  --width 2048 \
  --num-inference-steps 4 \
  --layer-ids all \
  --compression-rate 2 \
  --compressed-topk 64 \
  --seed 42 \
  --device cuda
