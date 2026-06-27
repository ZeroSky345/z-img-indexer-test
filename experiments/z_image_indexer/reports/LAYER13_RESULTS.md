# Z-Image Layer-13 Indexer Distillation Attempt

## Scope of This Step

This step is intentionally narrow.

It does **not** try multi-layer sparse replacement yet.

Instead, it prepares for that next step by training and validating a dedicated indexer for transformer layer `13`, because the previously trained indexer only targeted layer `12`.

## Setup

- Model: `Tongyi-MAI/Z-Image-Turbo`
- Experiment type: detached indexer distillation
- Target layer: `13`
- Resolution: `512x512`
- Scheduler setting: `4` steps
- Training steps: `1000`
- GPU: `NVIDIA A100 80GB PCIe`

## Result

From `summary_run_1000_steps_layer13.json`:

- initial loss: `0.6991989016532898`
- final loss: `0.10066670179367065`
- initial recall@64: `0.0596160888671875`
- final recall@64: `0.7633819580078125`

## Interpretation

This is a strong positive result.

The layer-13 indexer learned the teacher attention support even better than the earlier layer-12 experiment:

- layer 12 final recall@64: about `0.6157`
- layer 13 final recall@64: about `0.7634`

## What This Step Establishes

This step supports the idea that late layers beyond layer 12 also have learnable sparse support structure.

That means the next multi-layer sparse replacement experiment is now better grounded:

- layer 12 has a trained indexer
- layer 13 has a trained indexer

## Recommended Next Step

Use the layer-12 and layer-13 indexers together in a two-layer sparse generation comparison, and compare:

- output stability
- image similarity
- wall-clock time
