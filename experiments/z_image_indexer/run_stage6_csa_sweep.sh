#!/usr/bin/env bash
set -euo pipefail

source /tmp/diffsynth-venv/bin/activate
cd /tmp/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=0
export MODELSCOPE_CACHE=/tmp/modelscope-cache
export HF_HOME=/tmp/hf-home

for spec in \
  "m2_k64 2 64" \
  "m4_k96 4 96" \
  "m8_k64 8 64" \
  "m4_k48 4 48"
do
  set -- $spec
  name=$1
  m=$2
  k=$3
  echo "RUN:$name"
  python experiments/z_image_indexer/stage6_generate_compare_csa_single_layer.py \
    --model-base-path /tmp/DiffSynth-Studio/models \
    --indexer-ckpt /tmp/DiffSynth-Studio/experiments/z_image_indexer/results/run_1000_steps/indexer_distill.pt \
    --output-dir /tmp/DiffSynth-Studio/experiments/z_image_indexer/results/csa_sweep_$name \
    --height 512 \
    --width 512 \
    --num-inference-steps 4 \
    --layer-id 12 \
    --compression-rate "$m" \
    --compressed-topk "$k" \
    --num-prompts 2
done
