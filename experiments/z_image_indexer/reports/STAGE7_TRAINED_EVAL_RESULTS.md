# Z-Image Trained CSA Indexer Evaluation Results

## Scope

This report summarizes the first post-training evaluation of the formalized CSA indexer training path.

Training configuration:

- model: `Z-Image-Turbo`
- layer: `12`
- route: `CSA-like`
- `compression_rate=2`
- `compressed_topk=64`
- training mode: **indexer-only**
- training steps: `1000`

Evaluation configuration:

- resolution: `512x512`
- `4-step`
- prompt count: `8`

## Output location on training server

```text
/tmp/DiffSynth-Studio/experiments/z_image_indexer/results/eval_csa_layer12_m2_topk64
```

## Summary

From `summary_eval_csa_layer12_m2_topk64_p8.json`:

- mean dense time (s): `0.5992989302612841`
- mean sparse time (s): `0.5874725813046098`
- mean sparse/dense ratio: `1.0461394241002129`
- mean image MSE: `0.01754386251559481`

## Interpretation

This post-training evaluation suggests:

1. **Quality remains relatively stable**
   - average image MSE is moderate and consistent with the pre-training candidate range

2. **Speed is close to parity, but not clearly better**
   - average sparse/dense ratio is about `1.046x`
   - that means sparse is still slightly slower on average in this evaluation

3. **The training result is usable, but not yet a decisive speed win**
   - the route still looks viable
   - but formal training has not yet turned the candidate into a clearly faster production point

## What This Means

The current default training candidate:

- `layer12 + m=2 + topk=64`

is still the best structured path to continue with, but the project should currently treat it as:

- a **quality-stable / near-parity runtime** point
- not yet a robust speedup point

## Recommended Next Step

Before expanding training scale, the most useful next work would be one of:

1. compare this trained checkpoint directly against the pre-training baseline in one consolidated report
2. test a neighboring CSA point such as `m=4, topk=48` in the same trained setup
3. improve execution efficiency further before larger training investment
