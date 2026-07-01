# Stage 13: All-Layer Training Support

## What Changed

`train_csa_multilayer_indexer.py` now supports all main Z-Image DiT layers.

Supported layer syntax:

- `--layer-ids all`
- `--layer-ids 0-29`
- `--layer-ids 12,13`
- mixed explicit/range syntax such as `--layer-ids 0,4,8-13`

The script also adds:

- `--batch-size`
  - implemented as gradient accumulation over multiple prompt/timestep/latent samples per optimizer step
- multi-layer Q/K capture in one teacher forward pass
  - avoids replaying layers from the beginning once per target layer

## Why This Matters

The earlier multi-layer script could train selected layers such as `12,13`, but it was not practical for all-layer training because each target layer required a separate teacher capture pass.

The new path captures all requested layers during the same forward traversal up to the deepest selected layer. This makes `all` layer training feasible enough to test before moving to formal multi-GPU scale-up.

## Recommended All-Layer Smoke Run

```bash
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_bs4.sh
```

Default configuration:

- layers: `all`
- batch size: `4`
- `compression_rate=2`
- `compressed_topk=64`
- `rank=128`
- resolution: `512x512`
- steps: `1000`
- training mode: indexer-only

## Hardware Note

Start with a single 80GB GPU class machine.

If this run hits out-of-memory:

- first reduce `--batch-size` from `4` to `2`
- then reduce to `1` if needed
- only then move to multi-GPU, because this is still indexer-only training rather than base-model joint training

## Current Status

This stage adds code support and launch files. It does not claim a quality or speed improvement until the all-layer run and post-training evaluation complete.
