import json
import math
from pathlib import Path

import torch

from train_indexer_distill import (
    DEFAULT_PROMPTS,
    BilinearIndexer,
    build_noise_latents,
    build_pipe,
    capture_prompt_embedder,
    capture_teacher_qk,
    set_seed,
)


def load_prompts(prompt_file: str | None, fallback_count: int | None = None) -> list[str]:
    if prompt_file is None:
        prompts = list(DEFAULT_PROMPTS)
    else:
        prompts = [line.strip() for line in Path(prompt_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {prompt_file}")
    if fallback_count is not None:
        prompts = prompts[:fallback_count]
    return prompts


def cache_prompt_embeds(pipe, prompts: list[str]) -> dict[str, torch.Tensor]:
    return {prompt: capture_prompt_embedder(pipe, prompt) for prompt in prompts}


def build_compressed_pool(image_k: torch.Tensor, compression_rate: int) -> torch.Tensor:
    num_tokens, num_heads, head_dim = image_k.shape
    block_count = math.ceil(num_tokens / compression_rate)
    pad = block_count * compression_rate - num_tokens
    if pad > 0:
        image_k = torch.cat([image_k, image_k[-1:].expand(pad, -1, -1)], dim=0)
    return image_k.view(block_count, compression_rate, num_heads, head_dim).mean(dim=1)


def compress_teacher_probs(teacher_probs: torch.Tensor, compression_rate: int) -> torch.Tensor:
    # teacher_probs: [Q, K]
    q_len, k_len = teacher_probs.shape
    block_count = math.ceil(k_len / compression_rate)
    pad = block_count * compression_rate - k_len
    if pad > 0:
        teacher_probs = torch.cat([teacher_probs, torch.zeros((q_len, pad), dtype=teacher_probs.dtype, device=teacher_probs.device)], dim=1)
    block_probs = teacher_probs.view(q_len, block_count, compression_rate).sum(dim=-1)
    block_probs = block_probs / block_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return block_probs


def topk_recall(teacher_scores: torch.Tensor, student_scores: torch.Tensor, k: int) -> float:
    teacher_top = teacher_scores.topk(k, dim=-1).indices
    student_top = student_scores.topk(k, dim=-1).indices
    hit = 0.0
    for i in range(teacher_top.shape[0]):
        hit += len(set(teacher_top[i].tolist()) & set(student_top[i].tolist())) / k
    return hit / teacher_top.shape[0]


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
