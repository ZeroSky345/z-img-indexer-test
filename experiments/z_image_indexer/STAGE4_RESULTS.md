# Z-Image Stage-Four Two-Layer Sparse Generation Results

## Scope of This Step

This step is a small, controlled follow-up to the one-layer stage-three comparison.

It does only one new thing:

- replace **two** late layers instead of one
- specifically, sparse-replace layer `12` and layer `13`
- use the separately trained indexers for those two layers

Nothing else changes:

- same base model: `Z-Image-Turbo`
- same resolution: `512x512`
- same 4-step setup
- same `topk=256`
- same prompt-count: `2`

## Summary

From `summary_stage4_layers12_13_topk256.json`:

- mean dense time (s): `0.7838983628898859`
- mean sparse time (s): `1.7042097225785255`
- mean sparse/dense time ratio: `2.4609257990142335`
- mean image MSE: `0.02133586979471147`
- mean step recall: `0.6475377082824707`

For reference, the previous one-layer stage-three result was:

- mean sparse/dense time ratio: `1.5734720874375816`
- mean image MSE: `0.018657910637557507`
- mean step recall: `0.5445523262023926`

## Interpretation

This result is informative:

1. **Support quality improved**
   - mean step recall increased from about `0.5446` to about `0.6475`
   - that is consistent with layer 13 being highly learnable

2. **Output drift increased slightly**
   - mean image MSE increased from about `0.0187` to about `0.0213`
   - so quality stayed in the same rough range, but moved a bit further away from dense baseline

3. **Runtime got worse**
   - sparse/dense ratio worsened from about `1.57x` to about `2.46x`
   - this confirms that stacking more Python-level sparse replacement makes the current prototype slower, not faster

## What This Step Establishes

This step shows that:

- multi-layer sparse replacement is still technically viable
- adding a second late layer does not make the generation collapse
- but it increases overhead substantially in the current implementation

## Practical Conclusion

The two-layer result strengthens the same overall message from earlier stages:

- **quality-side feasibility remains real**
- **speed-side viability still does not hold in the current prototype**

In other words:

- the learned routing idea continues to look meaningful
- the current execution path is still the bottleneck

## Recommended Next Step

Do **not** keep stacking more raw-token Python sparse layers in the current form.

The more valuable next step is:

1. stop scaling raw-token replacement depth
2. move to `CSA`-style compressed image memory
3. or build a fused sparse execution path first

That is the only realistic route toward actual speed benefit.
