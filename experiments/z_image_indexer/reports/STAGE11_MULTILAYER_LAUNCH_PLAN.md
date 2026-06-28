# Stage 11: First Multi-Layer Launch Plan

## Goal

Turn the earlier multi-layer preparation into a concrete first-run plan.

This stage still does **not** launch the run. It answers:

1. what the first multi-layer run should be
2. whether it should remain single-GPU
3. what exact train/eval commands should be used
4. what stop conditions should be checked before and after launch

## First Multi-Layer Candidate

Recommended first multi-layer candidate:

- model: `Z-Image-Turbo`
- route: `CSA-like`
- layers: `12,13`
- `compression_rate=2`
- `compressed_topk=64`
- training mode: **indexer-only**
- resolution: `512x512`
- sampling steps: `4`

## Hardware Decision

### Recommended decision for the first multi-layer run

- **still start on a single 80GB GPU**

### Why single-GPU is still acceptable

- still indexer-only
- still `512x512`
- no base-model joint training
- no high-resolution schedule
- the goal of the first multi-layer run is validation, not throughput

### When multi-GPU should be considered instead

Switch to a multi-GPU plan if any of the following become true:

- the first single-GPU multi-layer run is too slow to iterate on
- the project moves beyond two layers
- resolution increases above `512x512`
- training schedule length is increased materially
- any base-model joint training is introduced

## Recommended First Launch Shape

### Training

- single GPU
- one output directory dedicated to the run
- keep all hyperparameters aligned with the current single-layer default except `layers=12,13`

### Evaluation

- use the same benchmark prompt set
- compare directly against:
  - trained single-layer `m=2, topk=64`
  - dense baseline

## Success Criteria

The first multi-layer run is promising only if all of the following are broadly true:

- output quality does not regress materially relative to the single-layer default
- sparse/dense runtime does not worsen enough to erase the value of the second layer
- the trained result still looks stable across the benchmark prompt set

## Failure Criteria

Treat the first multi-layer run as a stop signal if any of the following happens:

- large quality regression
- large runtime regression
- unstable or inconsistent behavior across prompts
- training becomes operationally too slow on single GPU

## Practical Recommendation

At this point, the project is ready for a **first controlled multi-layer run**,
but that run should still be treated as:

- a validation run
- on single GPU first
- with explicit post-run comparison against the single-layer default

Only after that run should the project decide whether multi-GPU scale-up is justified.
