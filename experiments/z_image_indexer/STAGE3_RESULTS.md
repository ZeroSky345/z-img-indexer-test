# Z-Image Indexer Stage-Three Full Generation Results

## Goal

Stage two only measured layer-level and denoiser-output approximation.

Stage three asks the practical question:

> if we actually run full 4-step image generation with one sparse-replaced late layer, what happens to output similarity and wall-clock time?

## Test Setup

- Model: `Tongyi-MAI/Z-Image-Turbo`
- Sparse layer: `12`
- Sparse setting: `topk=256`
- Resolution: `512x512`
- Sampling steps: `4`
- Prompts evaluated: `2`
- Dense baseline and sparse run use the same prompt and seed

## Summary

From `summary_stage3_topk256.json`:

- mean dense time (s): `0.7907182555645704`
- mean sparse time (s): `1.097807738929987`
- mean sparse/dense time ratio: `1.5734720874375816`
- mean image MSE: `0.018657910637557507`
- mean step recall: `0.5445523262023926`

Per prompt:

### Prompt 0

- dense time (s): `1.0689016338437796`
- sparse time (s): `1.119487438350916`
- time ratio: `1.047325032449646`
- image MSE: `0.008131831884384155`

### Prompt 1

- dense time (s): `0.5125348772853613`
- sparse time (s): `1.076128039509058`
- time ratio: `2.0996191424255173`
- image MSE: `0.029183989390730858`

## Interpretation

This stage gives a mixed result:

1. **Output similarity is still reasonably good**
   - the sparse run does not collapse
   - the dense and sparse images remain visibly related
   - image MSE stays moderate for this simple one-layer replacement

2. **The current prototype is not faster**
   - on average, the sparse prototype is slower than dense baseline
   - this is expected because the current implementation is a Python-level proof of concept, not a fused sparse kernel

## What This Means

The main conclusion is:

- **quality-side feasibility remains positive**
- **speed-side benefit is not demonstrated yet**

So the research answer is:

- yes, `indexer`-guided sparse replacement in `Z-Image` is technically viable
- no, this prototype does not yet deliver real acceleration

## Practical Reading

At this point, the route still looks promising only if you are willing to do the systems work:

- fused sparse gather / attention kernel
- better sequence packing
- possibly `CSA`-style compressed memory instead of raw token sparse routing

Without those optimizations, Python-level sparse replacement is not enough to beat the dense baseline.

## Recommended Next Step

If the goal is real speedup, the next step should be:

1. keep the learned indexer
2. move from raw token sparse routing toward `CSA`-style compressed image memory
3. replace Python-side sparse score masking with a real fused sparse execution path
