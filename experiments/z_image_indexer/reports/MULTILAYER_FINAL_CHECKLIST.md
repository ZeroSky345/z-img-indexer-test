# Multi-Layer Final Checklist

## Scope

This checklist is for the first real multi-layer training decision point.

Current proposed candidate:

- model: `Z-Image-Turbo`
- route: `CSA-like`
- layers: `12,13`
- `compression_rate=2`
- `compressed_topk=64`
- training mode: start with **indexer-only**

## A. Evidence Check

- [ ] single-layer raw-token experiments are complete
- [ ] single-layer CSA-like experiments are complete
- [ ] candidate sweep has identified a default point
- [ ] trained single-layer post-training evaluation exists
- [ ] 8-prompt or broader benchmark exists
- [ ] the team agrees the current default point is `m=2, topk=64`

## B. Decision Check

- [ ] there is a concrete reason to go beyond single-layer training
- [ ] the expected value of adding layer 13 is clear
- [ ] the goal is explicit:
  - [ ] quality improvement
  - [ ] speed improvement
  - [ ] both

## C. Hardware Check

Keep using **single GPU** only if:

- [ ] still indexer-only
- [ ] still `512x512`
- [ ] still moderate training length

Escalate to a multi-GPU plan if:

- [ ] two-layer training is too slow for practical iteration
- [ ] resolution will be increased above `512x512`
- [ ] schedule length will be increased materially
- [ ] any base-model joint training is introduced

## D. Launch Readiness Check

- [ ] multi-layer config file exists
- [ ] multi-layer launch template exists
- [ ] benchmark prompt file is fixed
- [ ] training output directory is reserved
- [ ] evaluation output directory is reserved
- [ ] summary / records / run_config format is fixed

## E. Stop Conditions

Do **not** launch the first multi-layer run yet if:

- [ ] current single-layer default still has unresolved execution issues
- [ ] benchmark quality is not considered stable enough
- [ ] hardware decision is still unclear

## F. Go / No-Go Meaning

### Go

All of the following are true:

- single-layer evidence is stable enough
- training target is frozen
- benchmark is frozen
- hardware choice is justified

### No-Go

Any of the following is still true:

- single-layer tradeoff is still changing materially
- the default candidate is not settled
- multi-layer launch would mix too many unknowns at once
