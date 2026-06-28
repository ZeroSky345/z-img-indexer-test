# Stage 12: First Multi-Layer Training Result

## Scope

This stage is the first actual multi-layer training run.

Configuration:

- model: `Z-Image-Turbo`
- route: `CSA-like`
- layers: `12,13`
- `compression_rate=2`
- `compressed_topk=64`
- training mode: **indexer-only**
- resolution: `512x512`
- steps: `1000`
- hardware: **single 80GB GPU class machine**

## Summary

From `results/stage9/summary_train_csa_layers12_13_m2_topk64.json`:

- `initial_loss = 0.5688043832778931`
- `final_loss = 0.6681196093559265`
- `initial_layer_recalls`
  - layer 12: `0.0260009765625`
  - layer 13: `0.06304931640625`
- `final_layer_recalls`
  - layer 12: `0.399658203125`
  - layer 13: `0.56915283203125`

## Interpretation

This run is a **technical success** because:

- the multi-layer training script ran successfully end-to-end
- both layers improved their recall substantially

However, the result is not as clean as the earlier single-layer runs:

- the final loss is not lower than the initial loss
- layer-level recalls improved, but not to the same level as the strongest single-layer training results

So the current reading is:

- multi-layer indexer-only training is feasible on a single GPU
- but the first run does **not yet** justify immediate scale-up

## Practical Conclusion

This stage moves the project past the “can we even launch a multi-layer run?” question.

That question is now answered:

- **yes, we can**
- **and we can still do it on a single GPU**

But the next critical question becomes:

- does multi-layer training actually improve the quality/speed frontier enough to be worth continuing?

That still needs post-training evaluation before any multi-GPU decision.

## Recommended Next Step

Run the first multi-layer post-training evaluation against:

- dense baseline
- trained single-layer `m=2, topk=64`

Only after that comparison should the project decide whether multi-layer scale-up is worth further investment.
