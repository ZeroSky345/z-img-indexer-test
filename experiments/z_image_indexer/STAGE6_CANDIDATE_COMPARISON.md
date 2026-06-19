# Z-Image CSA Candidate Comparison Before Large-Scale Training

## Goal

After the initial sweep and one expanded validation, the remaining question was:

> which single CSA-like point should become the most serious candidate before any larger-scale training?

The two candidates compared on the same 4-prompt setup are:

- `m=2, topk=64`
- `m=4, topk=48`

## Result Table

| Candidate | Mean Sparse/Dense Ratio | Mean Image MSE | Reading |
| --- | ---: | ---: | --- |
| `m=2, topk=64` | `0.9805637295239189` | `0.01834007352590561` | Near runtime parity, best quality of the two |
| `m=4, topk=48` | `0.9635257141621854` | `0.02085925464052707` | Slightly faster, slightly worse quality |

## Interpretation

This comparison makes the tradeoff concrete:

- `m=4, topk=48` is the more speed-oriented point
- `m=2, topk=64` is the more quality-oriented point

The difference is not huge, but the direction is clear:

1. **Speed**
   - `m=4, topk=48` is modestly faster
   - its sparse/dense ratio is lower (`0.9635x` vs `0.9806x`)

2. **Quality**
   - `m=2, topk=64` is clearly better on image MSE
   - it preserves output more faithfully (`0.01834` vs `0.02086`)

## Recommendation

Before large-scale training, the better default candidate is:

- **`m=2, topk=64`**

Reason:

- the speed difference between the two points is relatively small
- the quality difference is more meaningful
- the project is still in a stage where output stability matters more than squeezing out a few extra percent of runtime

## Practical Takeaway

The project now has a more grounded pre-training target:

- route family: **CSA-like compressed-memory sparse routing**
- preferred initial operating point: **`compression_rate=2`, `compressed_topk=64`**

If the project later shifts toward more aggressive speed optimization, then `m=4, topk=48` becomes the next alternative worth revisiting.
