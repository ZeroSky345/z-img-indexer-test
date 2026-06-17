# Z-Image Indexer Stage-Two Sparse Replacement Results

## Goal

Stage one only proved that a detached indexer could learn teacher attention support.

Stage two asks a stronger question:

> if we actually use that trained indexer to sparsify one late transformer layer, how much does the final denoiser output drift away from the dense baseline?

## Test Setup

- Model: `Tongyi-MAI/Z-Image-Turbo`
- Frozen teacher model
- Replaced layer: `12`
- Scope of replacement:
  - only image-token queries use sparse image-token key selection
  - non-image keys remain dense
  - non-image queries remain dense
- Resolution: `512x512`
- Scheduler setting: `4` steps
- Evaluated `topk`: `64`, `128`, `256`
- Evaluated over 4 prompts x 4 denoising steps

## Summary

From `summary_stage2_layer12.json`:

### topk = 64

- mean recall@k: `0.4764575958251953`
- mean attention-output mse: `88.890625`
- mean attention-output cosine: `0.839599609375`
- mean final noise-pred mse: `0.09711456298828125`
- mean final noise-pred cosine: `0.964111328125`

### topk = 128

- mean recall@k: `0.5366921424865723`
- mean attention-output mse: `43.765625`
- mean attention-output cosine: `0.916748046875`
- mean final noise-pred mse: `0.08448266983032227`
- mean final noise-pred cosine: `0.968994140625`

### topk = 256

- mean recall@k: `0.6123306751251221`
- mean attention-output mse: `17.068359375`
- mean attention-output cosine: `0.9658203125`
- mean final noise-pred mse: `0.05866503715515137`
- mean final noise-pred cosine: `0.978515625`

## Interpretation

This is another positive signal.

Using the trained indexer to sparsify one late layer did not catastrophically break the denoiser output. As `topk` increased from `64` to `256`, both layer-level approximation quality and final noise-prediction similarity improved consistently.

The `topk=256` setting is the most promising among the tested options:

- best recall
- lowest layer-output error
- lowest final denoiser-output error
- highest final cosine similarity to dense baseline

## What This Means

The second-stage test suggests that a one-layer sparse replacement path is plausible in `Z-Image`, at least for a frozen-model approximation experiment.

In other words:

- stage one showed the support is learnable
- stage two showed that using that learned support in one actual layer can keep the final denoiser output reasonably close to dense baseline

## What This Still Does Not Prove

This still does not prove:

1. end-to-end image quality is preserved
2. real wall-clock inference is faster
3. this is the best sparse pattern
4. more aggressive sparsification across multiple layers is safe

## Recommended Next Step

The next experiment should be a true inference-time replacement test:

- keep text tokens dense
- use image-token sparse routing only in one or two late layers
- generate full 4-step images
- compare latency and image quality against dense baseline
