# Z-Image Experiment Reports

## Purpose

This directory contains the human-readable writeups for the `Z-Image` sparse-routing experiments.

## Suggested Reading Order

1. `RESULTS.md`
   - earliest standalone indexer-distillation PoC summary
2. `STAGE2_RESULTS.md`
   - one-layer sparse replacement approximation results
3. `STAGE3_RESULTS.md`
   - first real full-generation single-layer sparse comparison
4. `STAGE4_RESULTS.md`
   - two-layer raw-token sparse comparison
5. `STAGE5_RESULTS.md`
   - vectorized execution-path improvement for the two-layer raw-token route
6. `STAGE6_RESULTS.md`
   - first minimal CSA-like compressed-memory result
7. `STAGE6_SWEEP_RESULTS.md`
   - small CSA-like parameter sweep
8. `STAGE6_EXPANDED_RESULTS.md`
   - expanded validation of the leading CSA-like point
9. `STAGE6_CANDIDATE_COMPARISON.md`
   - head-to-head comparison between the two most relevant CSA-like candidates
10. `STAGE7_TRAINED_EVAL_RESULTS.md`
   - post-training evaluation of the current default CSA-like candidate
11. `STAGE8_TRAINED_CSA_CANDIDATE_COMPARISON.md`
   - trained-point comparison between the main CSA-like candidates

## Operational Docs

- `TRAINING_QUICKSTART.md`
  - direct server-side start instructions for the current default training point
- `FORMAL_TRAINING_CHECKLIST.md`
  - pre-flight checklist before longer formal training runs
- `BENCHMARK_QUICKSTART.md`
  - standard benchmark launch path for comparison runs
- `STAGE9_MULTILAYER_PREP.md`
  - preparation notes before the first multi-layer training phase
- `MULTILAYER_FINAL_CHECKLIST.md`
  - final pre-launch checklist before any first multi-layer run
- `STAGE10_MULTILAYER_GO_NO_GO.md`
  - current decision summary on whether to cross into the multi-layer phase

## Additional Note

The current default candidate for the next formal training phase is still:

- `Z-Image-Turbo`
- `layer12`
- `compression_rate=2`
- `compressed_topk=64`
