# Z-Image Indexer Distillation Results

## Setup

- Model: `Tongyi-MAI/Z-Image-Turbo`
- Experiment type: detached indexer distillation
- Teacher: frozen dense attention
- Student: lightweight bilinear indexer
- Distilled target: image-token to image-token attention
- Scheduler setting: 4 steps
- Resolution: `512x512`
- Target layer: `12`
- Training steps: `1000`
- Server: `xfusion4`
- GPU: `NVIDIA A100 80GB PCIe`

## What Was Trained

Only the standalone indexer was updated.

The original `Z-Image-Turbo` model stayed frozen throughout the run.

## Metrics

From `summary_run_1000_steps.json`:

- initial loss: `0.5500845909118652`
- final loss: `0.4169244170188904`
- initial recall@64: `0.0548858642578125`
- final recall@64: `0.61572265625`

## Interpretation

This is a positive feasibility signal.

The standalone indexer learned a much better approximation to the teacher attention support than its random initialization. In particular, `recall@64` increased from about `5.5%` to about `61.6%`.

That means the attention support in this `Z-Image` configuration appears to be learnable by a cheap detached indexer.

## Important Limitation

This experiment does **not** prove end-to-end generation improvement.

It only shows that:

1. teacher attention support is learnable
2. a detached indexer can fit part of that structure quickly

It does **not** yet show:

1. faster inference
2. stable image quality after replacing dense attention
3. the best sparse pattern or the best layer schedule

## Recommended Next Step

Move to a second-stage experiment:

- keep text tokens dense
- try a `CSA`-style path on image tokens
- replace one or a few late transformer layers with `local dense + indexed global sparse`
- compare latency and image quality against the original 4-step model
