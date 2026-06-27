# Z-Image Indexer Experiments

## Structure

- `csa_common.py`
  - shared helpers for CSA-style training/evaluation
- `train_*.py`, `eval_*.py`, `stage*.py`
  - executable experiment and evaluation scripts
- `run_*.sh`
  - direct shell entrypoints for server-side execution
- `configs/`
  - recommended config snapshots and prompt files
- `reports/`
  - human-readable experiment conclusions and summaries
- `results/`
  - JSON metrics and summaries grouped by stage
- `comparisons/`
  - side-by-side dense vs sparse image comparisons grouped by stage

## Current default training candidate

- model: `Z-Image-Turbo`
- layer: `12`
- route: `CSA-like`
- `compression_rate=2`
- `compressed_topk=64`

See:

- `reports/TRAINING_QUICKSTART.md`
- `configs/config_recommended_m2_topk64.json`
