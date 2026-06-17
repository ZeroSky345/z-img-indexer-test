import argparse
import json
import math
import os
import random
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth.core import ModelConfig
from diffsynth.pipelines.z_image import (
    ZImagePipeline,
    ZImageUnit_InputImageEmbedder,
    ZImageUnit_NoiseInitializer,
    ZImageUnit_PromptEmbedder,
)
from einops import rearrange


DEFAULT_PROMPTS = [
    "Young Chinese woman in red Hanfu, intricate embroidery, soft-lit night scene, cinematic portrait.",
    "A futuristic poster with bold Chinese typography, bright neon colors, complex layout, high detail.",
    "A studio product shot of a porcelain tea set on a wooden table, soft natural light, ultra detailed.",
    "A fantasy city skyline at dusk, floating lanterns, intricate architecture, painterly but realistic.",
]


class CaptureStop(RuntimeError):
    pass


class BilinearIndexer(nn.Module):
    def __init__(self, input_dim: int, rank: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, rank, bias=False)
        self.k_proj = nn.Linear(input_dim, rank, bias=False)
        self.scale = rank ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        qh = self.q_proj(q)
        kh = self.k_proj(k)
        return torch.matmul(qh, kh.transpose(0, 1)) * self.scale


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--model-base-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--layer-id", type=int, default=12)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--recall-k", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pipe(args):
    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.model_base_path

    model_configs = [
        ModelConfig(model_id=args.model_id, origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id=args.model_id, origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ]
    tokenizer_config = ModelConfig(model_id=args.model_id, origin_file_pattern="tokenizer/")
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        enable_npu_patch=False,
    )
    pipe.scheduler.set_timesteps(args.num_inference_steps)
    pipe.requires_grad_(False)
    pipe.eval()
    return pipe


def capture_prompt_embedder(pipe, prompt: str):
    unit = ZImageUnit_PromptEmbedder()
    result = unit.process(pipe, prompt, None)
    return result["prompt_embeds"][0]


def build_noise_latents(pipe, height: int, width: int, seed: int):
    noise_unit = ZImageUnit_NoiseInitializer()
    input_unit = ZImageUnit_InputImageEmbedder()
    noise = noise_unit.process(pipe, height, width, seed, pipe.device)["noise"]
    latents = input_unit.process(pipe, None, noise)["latents"]
    return latents


def project_qk(attention_module, hidden_states, freqs_cis):
    query = attention_module.to_q(hidden_states)
    key = attention_module.to_k(hidden_states)

    query = query.unflatten(-1, (attention_module.num_heads, -1))
    key = key.unflatten(-1, (attention_module.num_heads, -1))

    if attention_module.norm_q is not None:
        query = attention_module.norm_q(query)
    if attention_module.norm_k is not None:
        key = attention_module.norm_k(key)

    if freqs_cis is not None:
        query = attention_module.apply_rotary_emb(query, freqs_cis)
        key = attention_module.apply_rotary_emb(key, freqs_cis)

    return query, key


def patchify_turbo_inputs(dit, latents, prompt_embeds, timestep):
    timestep = 1000 - timestep
    t_noisy = dit.t_embedder(timestep)

    latents = rearrange(latents, "B C H W -> C B H W")
    x, cap_feats, patch_metadata = dit.patchify_and_embed([latents], [prompt_embeds])
    x = x[0]
    cap_feats = cap_feats[0]

    x_real_len = int((~patch_metadata["x_pad_mask"][0]).sum().item())
    cap_real_len = int((~patch_metadata["cap_pad_mask"][0]).sum().item())

    x = dit.all_x_embedder["2-1"](x)
    x[patch_metadata["x_pad_mask"][0]] = dit.x_pad_token.to(dtype=x.dtype, device=x.device)
    x_freqs_cis = dit.rope_embedder(torch.cat(patch_metadata["x_pos_ids"], dim=0))
    x = rearrange(x, "L C -> 1 L C")
    x_freqs_cis = rearrange(x_freqs_cis, "L C -> 1 L C")

    for layer in dit.noise_refiner:
        x = layer(x=x, attn_mask=None, freqs_cis=x_freqs_cis, adaln_input=t_noisy)

    cap_feats = dit.cap_embedder(cap_feats)
    cap_feats[patch_metadata["cap_pad_mask"][0]] = dit.cap_pad_token.to(dtype=x.dtype, device=x.device)
    cap_freqs_cis = dit.rope_embedder(torch.cat(patch_metadata["cap_pos_ids"], dim=0))
    cap_feats = rearrange(cap_feats, "L C -> 1 L C")
    cap_freqs_cis = rearrange(cap_freqs_cis, "L C -> 1 L C")

    for layer in dit.context_refiner:
        cap_feats = layer(x=cap_feats, attn_mask=None, freqs_cis=cap_freqs_cis)

    unified = torch.cat([x, cap_feats], dim=1)
    unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)
    return unified, unified_freqs_cis, t_noisy, x_real_len, cap_real_len


def capture_teacher_qk(pipe, prompt_embeds, latents, timestep, layer_id):
    dit = pipe.dit
    unified, unified_freqs_cis, t_noisy, x_real_len, cap_real_len = patchify_turbo_inputs(
        dit, latents, prompt_embeds, timestep
    )

    captured = {}
    target_layer = dit.layers[layer_id]
    original_forward = target_layer.attention.forward

    def patched_forward(self, hidden_states, freqs_cis, attention_mask):
        q, k = project_qk(self, hidden_states, freqs_cis)
        captured["q"] = q.detach()
        captured["k"] = k.detach()
        raise CaptureStop

    target_layer.attention.forward = types.MethodType(patched_forward, target_layer.attention)
    try:
        for idx, layer in enumerate(dit.layers):
            unified = layer(x=unified, attn_mask=None, freqs_cis=unified_freqs_cis, adaln_input=t_noisy)
            if idx == layer_id:
                break
    except CaptureStop:
        pass
    finally:
        target_layer.attention.forward = original_forward

    if "q" not in captured:
        raise RuntimeError(f"Failed to capture Q/K for layer {layer_id}")

    return captured["q"], captured["k"], x_real_len, cap_real_len


def compute_teacher_scores(q, k):
    head_dim = q.shape[-1]
    return torch.einsum("qhd,khd->qkh", q, k).mean(dim=-1) / math.sqrt(head_dim)


def topk_recall(teacher_scores, student_scores, k: int):
    teacher_top = teacher_scores.topk(k, dim=-1).indices
    student_top = student_scores.topk(k, dim=-1).indices
    hit = 0.0
    for i in range(teacher_top.shape[0]):
        hit += len(set(teacher_top[i].tolist()) & set(student_top[i].tolist())) / k
    return hit / teacher_top.shape[0]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    pipe = build_pipe(args)
    prompt_cache = {prompt: capture_prompt_embedder(pipe, prompt) for prompt in DEFAULT_PROMPTS}

    sample_q, sample_k, x_real_len, _ = capture_teacher_qk(
        pipe=pipe,
        prompt_embeds=prompt_cache[DEFAULT_PROMPTS[0]],
        latents=build_noise_latents(pipe, args.height, args.width, args.seed),
        timestep=pipe.scheduler.timesteps[0].unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype),
        layer_id=args.layer_id,
    )
    input_dim = sample_q[0, :x_real_len].flatten(1).shape[-1]

    indexer = BilinearIndexer(input_dim=input_dim, rank=args.rank).to(pipe.device)
    optimizer = torch.optim.AdamW(indexer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    start = time.time()

    for step in range(1, args.steps + 1):
        prompt = random.choice(DEFAULT_PROMPTS)
        prompt_embeds = prompt_cache[prompt]
        timestep = random.choice(pipe.scheduler.timesteps).unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype)
        latents = build_noise_latents(pipe, args.height, args.width, args.seed + step)

        with torch.no_grad():
            q, k, x_real_len, _ = capture_teacher_qk(pipe, prompt_embeds, latents, timestep, args.layer_id)
            q = q[0, :x_real_len].float()
            k = k[0, :x_real_len].float()
            teacher_scores = compute_teacher_scores(q, k)
            teacher_probs = F.softmax(teacher_scores, dim=-1)

        q_flat = q.flatten(1)
        k_flat = k.flatten(1)
        student_scores = indexer(q_flat, k_flat)
        student_log_probs = F.log_softmax(student_scores, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            recall = topk_recall(teacher_scores, student_scores, args.recall_k)
            entropy = -(student_log_probs.exp() * student_log_probs).sum(dim=-1).mean().item()

        record = {
            "step": step,
            "loss": float(loss.item()),
            "recall_at_k": float(recall),
            "student_entropy": float(entropy),
        }
        history.append(record)

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"step={step} loss={record['loss']:.6f} recall@{args.recall_k}={record['recall_at_k']:.4f} "
                f"entropy={record['student_entropy']:.4f} elapsed={elapsed:.1f}s"
            )

    torch.save(
        {
            "model_state_dict": indexer.state_dict(),
            "args": vars(args),
            "history": history,
        },
        os.path.join(args.output_dir, "indexer_distill.pt"),
    )
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    summary = {
        "steps": args.steps,
        "layer_id": args.layer_id,
        "height": args.height,
        "width": args.width,
        "recall_at_k": args.recall_k,
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "initial_recall_at_k": history[0]["recall_at_k"],
        "final_recall_at_k": history[-1]["recall_at_k"],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
