# Stage 10: Multi-Layer Go / No-Go

## Goal

Decide whether the project should now cross from the completed single-layer phase into the first real multi-layer training phase.

## Current State

The project has already completed:

- single-layer `DSA-like` validation
- single-layer `CSA-like` validation
- CSA-like parameter sweep
- expanded benchmark evaluation
- formalized training/eval entrypoints
- post-training comparison between the two leading CSA-like candidates

The current default single-layer point is:

- `layer12`
- `compression_rate=2`
- `compressed_topk=64`

## Current Default Point Reading

After training and broader evaluation, the current default point is best understood as:

- quality-stable
- near dense runtime
- not yet a robust speedup point

## Multi-Layer Candidate

First multi-layer target if the project moves forward:

- `layers = 12,13`
- `compression_rate = 2`
- `compressed_topk = 64`
- start with **indexer-only**

## Recommendation

### Decision

- **Not yet a full Go for immediate multi-layer launch**
- **Go for final preparation**

## Why

The project is now close to the boundary, but one thing is still true:

- the single-layer default is good enough to continue
- but it has not yet produced a clearly positive speed margin

That means the project should not jump into a more expensive multi-layer phase casually.

Instead, the right interpretation is:

- the project has enough evidence to prepare the multi-layer path
- but the first real multi-layer run should be treated as a deliberate next phase transition, not as a routine extension

## Practical Outcome

At this stage, the correct next move is:

1. freeze the benchmark and reporting format
2. freeze the multi-layer candidate config
3. explicitly decide whether the first multi-layer run remains single-GPU or needs multi-GPU

Only after that should the project actually launch the first multi-layer training.

## Final Guidance

If the user wants to continue immediately, the next deliverable should be:

- a final multi-layer launch plan
- including a hardware decision

Then, and only then, launch the first multi-layer run.
