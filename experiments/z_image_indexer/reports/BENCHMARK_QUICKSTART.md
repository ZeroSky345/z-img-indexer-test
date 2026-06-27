# Benchmark Quickstart

## Goal

This document defines the standard benchmark entrypoint for the current default CSA-like candidate:

- model: `Z-Image-Turbo`
- layer: `12`
- route: `CSA-like`
- `compression_rate=2`
- `compressed_topk=64`

The purpose is to make all future quality/runtime comparisons use the same prompt set and launch path.

## Benchmark Prompt Set

Default benchmark prompt file:

```text
experiments/z_image_indexer/configs/benchmark_prompts_v1.txt
```

This prompt set currently covers:

- portrait
- poster / typography
- product photo
- fantasy city
- modern city / landscape
- rainy street
- mountain / lake landscape
- cyberpunk city
- architecture / interior
- luxury product
- alpine village
- coastal skyline

## Recommended Launch

### Dense vs trained default candidate

```bash
python experiments/z_image_indexer/eval_csa_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
  --prompt-file experiments/z_image_indexer/configs/benchmark_prompts_v1.txt \
  --indexer-ckpt experiments/z_image_indexer/results/train_csa_layer12_m2_topk64/csa_indexer_distill.pt \
  --output-dir experiments/z_image_indexer/results/benchmark_eval_csa_layer12_m2_topk64_v1 \
  --height 512 \
  --width 512 \
  --num-inference-steps 4 \
  --layer-id 12 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --seed 42 \
  --device cuda
```

### Shell wrapper

```bash
bash experiments/z_image_indexer/run_eval_benchmark_m2_topk64.sh
```

## Outputs

```text
experiments/z_image_indexer/results/benchmark_eval_csa_layer12_m2_topk64_v1/
├─ records.json
├─ summary.json
├─ run_config.json
├─ prompt_0_dense.png
├─ prompt_0_csa.png
├─ prompt_0_compare_csa.png
└─ ...
```

## Usage Rule

For any future comparison that is meant to influence training direction:

- prefer this benchmark prompt file
- prefer this launch configuration
- do not silently change prompt composition mid-comparison
