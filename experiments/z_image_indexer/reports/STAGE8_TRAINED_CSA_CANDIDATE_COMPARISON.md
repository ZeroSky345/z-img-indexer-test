# Z-Image Trained CSA Candidate Comparison

## Goal

After formalized indexer-only training and post-training evaluation, compare the two main CSA-like candidates under the same 8-prompt setup:

- `m=2, topk=64`
- `m=4, topk=48`

## Post-Training Result Table

| Candidate | Mean Sparse/Dense Ratio | Mean Image MSE | Reading |
| --- | ---: | ---: | --- |
| `m=2, topk=64` | `1.0461394241002129` | `0.01754386251559481` | Slightly slower than dense, better quality |
| `m=4, topk=48` | `1.0177750021356342` | `0.018816620606230572` | Slightly faster than the `m=2` trained point, slightly worse quality |

## Interpretation

After training, both candidates converge to a very similar runtime-quality region:

- neither one is a decisive speed win
- both are close to dense runtime
- both keep image deviation in a relatively modest range

The difference between them is now:

1. **Speed**
   - `m=4, topk=48` is the slightly more speed-oriented point
2. **Quality**
   - `m=2, topk=64` remains the slightly more quality-oriented point

## Recommendation

If forced to choose one default post-training point, the recommendation still stays with:

- **`m=2, topk=64`**

Reason:

- the speed gap is small
- the quality advantage is still meaningful
- it remains the safer candidate for further scaling and future integration work

## Practical Conclusion

This comparison does not overturn the earlier recommendation.

It refines it:

- `m=4, topk=48` remains a valid speed-leaning alternative
- `m=2, topk=64` remains the better balanced default candidate
