#!/usr/bin/env bash
set -euo pipefail

# Recommended LoRA adapter-weight training for Z-Image-Turbo.
# It trains only the new LoRA adapter weights while loading the public
# zimage_turbo_training_adapter as a preset LoRA to better preserve few-step behavior.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/tmp/modelscope-cache}"
export HF_HOME="${HF_HOME:-/tmp/hf-home}"

DATA_ROOT="${DATA_ROOT:-data/diffsynth_example_dataset}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/z_image/Z-Image-Turbo}"
ADAPTER_DIR="${ADAPTER_DIR:-models/ostris/zimage_turbo_training_adapter}"
PRESET_LORA_PATH="${PRESET_LORA_PATH:-${ADAPTER_DIR}/zimage_turbo_training_adapter_v1.safetensors}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/lora/z_image_turbo_lora_2048_adapter}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

modelscope download \
  --dataset DiffSynth-Studio/diffsynth_example_dataset \
  --include "z_image/Z-Image-Turbo/*" \
  --local_dir "${DATA_ROOT}"

modelscope download \
  --model ostris/zimage_turbo_training_adapter \
  --local_dir "${ADAPTER_DIR}"

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
  --preset_lora_path "${PRESET_LORA_PATH}" \
  --preset_lora_model "dit" \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 8

echo "LoRA adapter checkpoints were written to: ${OUTPUT_PATH}"
echo "Use the selected checkpoint as TEACHER_LORA_PATH, for example:"
echo "  export TEACHER_LORA_PATH=${OUTPUT_PATH}/epoch-4.safetensors"
