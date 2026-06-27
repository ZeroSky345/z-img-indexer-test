# Z-Image Stage-Six Expanded Validation Results

## Scope of This Step

This step does not introduce a new model path.

It only expands validation for the current best CSA-like candidate:

- layer `12`
- compression rate `2`
- compressed top-k `64`

The earlier result used only `2` prompts.

This step reruns the same configuration on `4` prompts to check whether the initial speed win is stable.

## Summary

### Earlier 2-prompt result

- mean sparse/dense ratio: `0.8492945368579132`
- mean image MSE: `0.028419848531484604`

### Expanded 4-prompt result

- mean dense time (s): `0.6537281991913915`
- mean sparse time (s): `0.5837888494133949`
- mean sparse/dense ratio: `0.9805637295239189`
- mean image MSE: `0.01834007352590561`

## Interpretation

This expanded validation changes the reading of the earlier result.

### Speed

The average speed advantage becomes much smaller on 4 prompts:

- still slightly faster than dense on average
- but only by about `1.9%`

So the original 2-prompt speed win was real, but optimistic.

### Quality

Image deviation improves materially on the larger prompt set:

- MSE drops from about `0.0284` to about `0.0183`

That means this configuration looks more stable on a slightly broader prompt distribution than the initial 2-prompt run suggested.

## What This Step Establishes

This step gives a more conservative and more trustworthy conclusion:

- `m=2, topk=64` is still the most balanced CSA-like candidate seen so far
- but its speed gain should currently be treated as **marginal**, not large
- its quality retention now looks stronger than the earlier small-sample estimate

## Practical Conclusion

After the expanded validation, the best interpretation is:

- `m=2, topk=64` remains the right next candidate to continue with
- but the project should no longer assume that it already has a robust speed win
- it now looks more like a **near-parity runtime point with relatively good quality retention**

## Recommended Next Step

Before large-scale training, the next most useful work is:

1. run one more moderate-size prompt expansion or a small fixed benchmark set
2. compare `m=2, topk=64` directly against `m=4, topk=48`
3. then decide which single CSA-like point becomes the official training target
