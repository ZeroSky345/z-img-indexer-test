# Stage 9: Multi-Layer Training Preparation

## Goal

This stage does **not** start multi-layer or multi-GPU training yet.

Its purpose is to prepare the project for that transition by answering:

1. what the first multi-layer training target should be
2. when single-GPU stops being the right choice
3. what configuration and launch shape should be used when the project crosses into the multi-layer stage

## Recommended First Multi-Layer Candidate

If the project enters multi-layer training, the first candidate should be:

- model: `Z-Image-Turbo`
- route: `CSA-like`
- layers: `12,13`
- `compression_rate=2`
- `compressed_topk=64`
- training mode: start with **indexer-only**

Reason:

- `layer12` is the most stable default point so far
- `layer13` also showed strong learnability in the distillation phase
- the two-layer sparse generation path already proved technically viable

## Why Multi-Layer Is the Next Real Step

The current single-layer path has already answered most of the low-cost questions:

- sparse routing is learnable
- compressed-memory routing is better than raw-token sparse
- a formalized single-layer training/eval loop now exists

The next question is no longer “can this work at all?”

It is:

> does adding a second trained sparse layer improve the speed/quality frontier enough to justify more engineering and hardware complexity?

## Hardware Decision Boundary

The project should continue using **single GPU** while all of the following remain true:

- one sparse layer only
- indexer-only training
- `512x512`
- moderate training length

The project should **seriously consider multi-GPU** when one or more of the following become true:

- training two or more layers together
- increasing image resolution materially above `512x512`
- running much longer schedules where wall-clock becomes dominant
- adding any joint adaptation of the base model

## Recommendation Before Switching to Multi-GPU

Before actually switching hardware strategy, finish these single-GPU preparations:

1. finalize the multi-layer training config
2. finalize the benchmark prompt set and post-training evaluation entry
3. define the exact success metrics for multi-layer comparison
4. confirm the checkpoint/output layout

That way the first multi-GPU run does not mix infrastructure churn with modeling uncertainty.

## Proposed Multi-Layer Success Metrics

When multi-layer training begins, compare the result directly against the current single-layer default:

- sparse/dense runtime ratio
- image MSE
- LPIPS if added later
- qualitative prompt coverage
- stability across multiple prompts and seeds

The key decision should be:

> does multi-layer training improve the overall quality/speed tradeoff enough to justify the extra complexity?

## Practical Next Step

The next implementation step should be **configuration preparation**, not immediate launch:

- create a formal multi-layer config template
- create a matching multi-layer launch template
- keep it disabled by default until the user explicitly approves the first multi-layer run
