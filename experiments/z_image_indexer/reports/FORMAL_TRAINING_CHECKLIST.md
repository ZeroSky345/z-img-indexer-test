# Formal Training Checklist

## Scope

This checklist is for the current recommended default training candidate:

- model: `Z-Image-Turbo`
- layer: `12`
- route: `CSA-like`
- `compression_rate=2`
- `compressed_topk=64`
- training mode: **indexer-only**

Use this checklist before starting any longer or more expensive training run.

## A. Repository / Code Check

- [ ] `experiments/z_image_indexer/train_csa_indexer.py` exists
- [ ] `experiments/z_image_indexer/eval_csa_indexer.py` exists
- [ ] `experiments/z_image_indexer/csa_common.py` exists
- [ ] `experiments/z_image_indexer/configs/default_prompts_train.txt` exists
- [ ] `experiments/z_image_indexer/configs/config_recommended_m2_topk64.json` exists
- [ ] `experiments/z_image_indexer/run_train_csa_m2_topk64.sh` exists
- [ ] `experiments/z_image_indexer/run_eval_csa_m2_topk64.sh` exists

## B. Environment Check

- [ ] GPU is visible: `nvidia-smi`
- [ ] Python environment can import the runtime dependencies
- [ ] `Z-Image-Turbo` can already run inference in this environment
- [ ] `CUDA_VISIBLE_DEVICES` is set appropriately
- [ ] `MODELSCOPE_CACHE` is set
- [ ] `HF_HOME` is set

Recommended:

```bash
export CUDA_VISIBLE_DEVICES=0
export MODELSCOPE_CACHE=/tmp/modelscope-cache
export HF_HOME=/tmp/hf-home
```

## C. Model / Path Check

- [ ] model base path is known
- [ ] tokenizer path is available through the default model path
- [ ] output directory does not conflict with an unrelated prior run

Recommended model path:

```bash
/tmp/DiffSynth-Studio/models
```

Recommended training output path:

```bash
experiments/z_image_indexer/results/train_csa_layer12_m2_topk64
```

Recommended evaluation output path:

```bash
experiments/z_image_indexer/results/eval_csa_layer12_m2_topk64
```

## D. Configuration Check

- [ ] `layer_id = 12`
- [ ] `compression_rate = 2`
- [ ] `compressed_topk = 64`
- [ ] `height = 512`
- [ ] `width = 512`
- [ ] `num_inference_steps = 4`
- [ ] `steps = 1000`
- [ ] `rank = 128`
- [ ] `lr = 1e-3`
- [ ] `weight_decay = 0.0`
- [ ] `recall_k = 16`
- [ ] `seed = 42`

## E. Prompt / Eval Check

- [ ] prompt file is fixed before training
- [ ] prompt file is reused in evaluation
- [ ] benchmark prompt set is not edited mid-run

Default file:

```bash
experiments/z_image_indexer/configs/default_prompts_train.txt
```

## F. Training Launch Check

- [ ] output directory exists or can be created
- [ ] there is enough disk space for logs, checkpoint, and eval outputs
- [ ] no conflicting process is already writing to the same output path

Recommended launch:

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64.sh
```

## G. Post-Training Check

After training completes, confirm:

- [ ] `csa_indexer_distill.pt` exists
- [ ] `metrics.json` exists
- [ ] `summary.json` exists
- [ ] `run_config.json` exists

## H. Evaluation Check

Recommended launch:

```bash
bash experiments/z_image_indexer/run_eval_csa_m2_topk64.sh
```

After evaluation completes, confirm:

- [ ] `records.json` exists
- [ ] `summary.json` exists
- [ ] `run_config.json` exists
- [ ] dense / sparse image outputs exist
- [ ] comparison images exist

## I. Decision Check

Before moving to a larger run, confirm:

- [ ] output quality is acceptable
- [ ] runtime is at least near parity with dense baseline
- [ ] the result still supports `m=2, topk=64` as the default point
- [ ] no major regression is visible relative to previous pre-training experiments

## J. Do Not Expand Yet Unless Needed

Do not immediately switch to:

- multi-layer training
- joint training with the base model
- higher resolution
- longer training schedules

unless this first formalized run is clearly stable and worth expanding.
