# Z-Image Stage-Five Vectorized Two-Layer Sparse Generation Results

## Scope of This Step

This step does not change the sparse routing policy.

It only changes the execution path:

- still sparse-replace layer `12` and layer `13`
- still use `topk=256`
- still use the same two trained indexers
- still compare against dense baseline with the same prompts and seeds

The only difference is:

- replace the previous Python-style sparse score construction with a more vectorized candidate gather + batched attention path

## Summary

From `summary_stage5_layers12_13_topk256_vectorized.json`:

- mean dense time (s): `0.8032714715227485`
- mean sparse time (s): `1.5927351685240865`
- mean sparse/dense time ratio: `2.2690952780202416`
- mean image MSE: `0.02174254390411079`
- mean step recall: `0.6477775573730469`

For reference, the previous stage-four two-layer result was:

- mean sparse/dense time ratio: `2.4609257990142335`
- mean image MSE: `0.02133586979471147`
- mean step recall: `0.6475377082824707`

## Interpretation

This step produced a modest but real execution improvement:

1. **Runtime improved somewhat**
   - sparse/dense ratio improved from about `2.46x` to about `2.27x`
   - so the vectorized execution path is better than the previous Python-heavy version

2. **Routing quality stayed effectively the same**
   - mean step recall remained about the same (`0.6475` -> `0.6478`)

3. **Output similarity stayed in the same range**
   - mean image MSE changed only slightly (`0.02134` -> `0.02174`)

## What This Step Establishes

This step confirms that:

- execution-path optimization does matter
- the current bottleneck is indeed in how sparse attention is executed, not only in the learned routing idea
- even a moderate vectorization pass can improve runtime without changing the sparse policy itself

## Practical Conclusion

This is a successful attempt, but not yet a complete win.

- It improves over the previous two-layer prototype
- But it is still much slower than dense baseline

So the updated conclusion is:

- learned sparse routing remains viable
- execution engineering can recover some performance
- but a larger systems jump is still required before sparse execution can become competitive

## Recommended Next Step

If continuing on the execution path, the next work should focus on:

1. reducing repeated tensor expansion / gather overhead further
2. moving toward a fused sparse gather-attention path
3. or changing the sparse unit from raw token routing to `CSA`-style compressed memory
