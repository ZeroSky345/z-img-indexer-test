# Z-Image Stage-Six CSA-Like Sweep Results

## Scope of This Step

This step keeps the model scope fixed and only scans a small CSA-like tradeoff grid on:

- layer `12`
- 4-step generation
- `512x512`
- 2 prompts

The tested configurations were:

- `m=2, topk=64`
- `m=4, topk=96`
- `m=8, topk=64`
- `m=4, topk=48`

We also compare against the earlier single-layer raw-token sparse baseline:

- stage three raw-token sparse: `topk=256`

## Sweep Table

| Config | Mean Sparse/Dense Ratio | Mean Image MSE | Notes |
| --- | ---: | ---: | --- |
| Raw-token baseline (`topk=256`) | `1.5735x` | `0.01866` | Slower than dense, best quality among sparse variants seen earlier |
| `m=2, topk=64` | `0.8493x` | `0.02842` | Best quality among the speed-winning CSA-like points |
| `m=4, topk=96` | `0.8905x` | `0.03553` | Speed win, but weaker quality than `m=2, topk=64` |
| `m=8, topk=64` | `0.8370x` | `0.03727` | Better speed, worse quality |
| `m=4, topk=48` | `0.8264x` | `0.03327` | Best speed in this sweep, mid-quality |

## Interpretation

This sweep confirms three things:

1. **CSA-like compressed memory is materially better than the earlier raw-token sparse path**
   - every tested compressed-memory point is faster than dense baseline on average
   - raw-token sparse was slower than dense baseline

2. **There is now a real speed/quality frontier**
   - lower ratio is better for speed
   - lower image MSE is better for quality
   - the configs do not all dominate each other

3. **`m=2, topk=64` looks like the best next training candidate**
   - it keeps a real speedup
   - it has the lowest image MSE among the speed-winning compressed-memory settings

## Practical Ranking

### Best quality among speed-winning points

- `m=2, topk=64`

### Best raw speed

- `m=4, topk=48`

### Best balanced candidate for next-step investment

- `m=2, topk=64`

## Recommendation

If choosing just one CSA-like configuration to continue with next, use:

- **`compression_rate=2`**
- **`compressed_topk=64`**

Reason:

- it is already faster than dense baseline on average
- it preserves image similarity better than the other compressed-memory sweep points

## What This Step Establishes

Before large-scale training, the project now has a much clearer answer:

- the route worth continuing is no longer raw-token sparse routing
- the route worth continuing is **compressed-memory sparse routing**
- and the current best candidate point is **`m=2, topk=64`**
