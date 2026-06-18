# Z-Image Stage-Six Minimal CSA-Style Single-Layer Results

## Scope of This Step

This step is the first move away from raw-token sparse routing toward a compressed-memory idea.

It keeps the experiment intentionally small:

- only one sparse-replaced layer: `12`
- fixed compression rate: `4`
- fixed compressed top-k: `64`
- same 4-step generation setup
- same 2 prompts as earlier stage-three comparisons

This is a **minimal CSA-style prototype**, not a full DeepSeek-style CSA implementation.

## Summary

From `summary_stage6_csa_layer12_m4_topk64.json`:

- mean dense time (s): `0.8029889510944486`
- mean sparse time (s): `0.5984960217028856`
- mean sparse/dense time ratio: `0.8412031283959656`
- mean image MSE: `0.032602082937955856`

For reference, the previous raw-token one-layer stage-three result was:

- mean sparse/dense time ratio: `1.5734720874375816`
- mean image MSE: `0.018657910637557507`

## Interpretation

This is the first experiment in this project that shows a real average speed win.

### Positive signal

- sparse runtime is now **faster** than dense baseline on average
- the mean sparse/dense ratio is about `0.84x`
- that corresponds to a speedup of roughly `15.9%`

### Tradeoff

- image deviation increased compared with the raw-token one-layer baseline
- mean image MSE rose from about `0.0187` to about `0.0326`

## What This Step Establishes

This step strongly supports the idea that:

- compressed-memory sparse routing is a better fit for `Z-Image` than the earlier raw-token sparse prototype
- moving toward a `CSA`-style design can trade some fidelity for real speed gains

In other words:

- raw-token routing proved feasibility
- compressed-memory routing is the first path that begins to show engineering payoff

## Practical Conclusion

This is a successful attempt.

It is important because it changes the project status from:

- “sparse routing works, but is slower”

to:

- “a minimal CSA-like compression design can already beat dense runtime, although quality tradeoffs remain”

## Recommended Next Step

The natural continuation is:

1. tune the compression / selection tradeoff
   - for example test `m=4, topk=96` or `m=2, topk=128`
2. compare multiple compressed-memory settings against this first speed-winning point
3. decide whether the quality/speed frontier is good enough to justify a more serious CSA implementation
