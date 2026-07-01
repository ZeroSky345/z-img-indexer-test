#!/usr/bin/env bash
set -euo pipefail

# Z-Image-Turbo DiT LoRA training at 2048x2048 with the official training adapter.
# This is the recommended LoRA path for Turbo if preserving few-step behavior matters.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/tmp/modelscope-cache}
export HF_HOME=${HF_HOME:-/tmp/hf-home}

DATA_ROOT=${DATA_ROOT:-data/diffsynth_example_dataset}
DATASET_DIR=${DATASET_DIR:-${DATA_ROOT}/z_image/Z-Image-Turbo}
OUTPUT_PATH=${OUTPUT_PATH:-./models/train/Z-Image-Turbo_lora_2048_adapter}
ADAPTER_PATH=${ADAPTER_PATH:-models/ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v1.safetensors}
NUM_PROCESSES=${NUM_PROCESSES:-4}

modelscope download \
  --dataset DiffSynth-Studio/diffsynth_example_dataset \
  --include "z_image/Z-Image-Turbo/*" \
  --local_dir "${DATA_ROOT}"

modelscope download \
  --model ostris/zimage_turbo_training_adapter \
  --include "zimage_turbo_training_adapter_v1.safetensors" \
  --local_dir "models/ostris/zimage_turbo_training_adapter"

accelerate launch --num_processes "${NUM_PROCESSES}" examples/z_image/model_training/train.py \
  --dataset_base_path "${DATASET_DIR}" \
  --dataset_metadata_path "${DATASET_DIR}/metadata.csv" \
  --height 2048 \
  --width 2048 \
  --max_pixels 4194304 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --preset_lora_path "${ADAPTER_PATH}" \
  --preset_lora_model "dit" \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 8
