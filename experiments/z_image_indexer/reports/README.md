# Z-Image Indexer Distillation PoC

## Goal

Test whether a lightweight standalone indexer can learn the image-token attention pattern of a frozen `Z-Image-Turbo` model.

This is a feasibility experiment, not a production sparse-attention replacement.

## Scope

- Freeze the original `Z-Image-Turbo` model.
- Use the 4-step scheduler setting.
- Distill one selected transformer layer at a time.
- Distill only image-token to image-token attention.
- Keep text tokens out of the learned routing path for this first PoC.

## Why This Design

`Z-Image` is not an LLM-style causal decode model. It recomputes the transformer stack on every denoising step, so the right first question is:

> can a cheap detached indexer predict the teacher attention support well enough to justify a later sparse-attention integration?

This PoC answers that question with the minimum possible engineering risk.

## Teacher / Student Setup

### Teacher

- Frozen `Z-Image-Turbo`
- Full attention
- Capture Q/K at one chosen unified-sequence transformer layer

### Student

- A lightweight bilinear indexer
- Input: detached teacher Q and K for image tokens only
- Output: dense score matrix over image-token keys

## Training Target

For each query image token:

- Compute teacher attention logits from teacher Q/K
- Average over heads
- Apply softmax over image-token keys
- Train the indexer with KL divergence against that distribution

## Metrics

- `kl`: KL divergence from student distribution to teacher distribution
- `recall_at_k`: overlap between teacher top-k keys and student top-k keys
- `student_entropy`: optional sanity signal for collapse

## Success Criteria

The PoC is considered technically promising if, within a bounded run:

- `kl` trends down
- `recall_at_k` trends up materially above initialization

This still does **not** prove end-to-end image quality or speed improvement. It only proves that the sparse support may be learnable.

## What This PoC Does Not Prove

- Faster inference
- Better image quality
- Safe replacement of full attention in all layers / all steps

Those require a second-stage experiment that actually swaps part of the attention path.
