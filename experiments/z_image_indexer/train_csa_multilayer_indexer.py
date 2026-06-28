import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from csa_common import (
    BilinearIndexer,
    build_compressed_pool,
    build_noise_latents,
    build_pipe,
    cache_prompt_embeds,
    capture_teacher_qk,
    compress_teacher_probs,
    load_prompts,
    set_seed,
    topk_recall,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--model-base-path", type=str, default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--layer-ids", type=str, default="12,13")
    parser.add_argument("--compression-rate", type=int, default=2)
    parser.add_argument("--compressed-topk", type=int, default=64)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--recall-k", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_layer_ids(value: str) -> list[int]:
    layer_ids = [int(v) for v in value.split(",") if v.strip()]
    if not layer_ids:
        raise ValueError("layer-ids cannot be empty")
    return layer_ids


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    layer_ids = parse_layer_ids(args.layer_ids)

    pipe = build_pipe(args)
    prompts = load_prompts(args.prompt_file)
    prompt_cache = cache_prompt_embeds(pipe, prompts)

    # Build one indexer per layer based on compressed key shape.
    indexers = {}
    for layer_id in layer_ids:
        sample_q, sample_k, x_real_len, _ = capture_teacher_qk(
            pipe=pipe,
            prompt_embeds=prompt_cache[prompts[0]],
            latents=build_noise_latents(pipe, args.height, args.width, args.seed),
            timestep=pipe.scheduler.timesteps[0].unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype),
            layer_id=layer_id,
        )
        sample_k = sample_k[0, :x_real_len].float()
        compressed_k = build_compressed_pool(sample_k, args.compression_rate)
        input_dim = compressed_k.flatten(1).shape[-1]
        indexers[layer_id] = BilinearIndexer(input_dim=input_dim, rank=args.rank).to(pipe.device)

    optimizer = torch.optim.AdamW(
        params=[p for module in indexers.values() for p in module.parameters()],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    start = time.time()

    for step in range(1, args.steps + 1):
        prompt = random.choice(prompts)
        prompt_embeds = prompt_cache[prompt]
        timestep = random.choice(pipe.scheduler.timesteps).unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype)
        latents = build_noise_latents(pipe, args.height, args.width, args.seed + step)

        layer_losses = []
        layer_recalls = {}
        layer_entropies = {}

        for layer_id in layer_ids:
            with torch.no_grad():
                q, k, x_real_len, _ = capture_teacher_qk(pipe, prompt_embeds, latents, timestep, layer_id)
                q = q[0, :x_real_len].float()
                k = k[0, :x_real_len].float()
                teacher_scores = torch.einsum("qhd,khd->qkh", q, k).mean(dim=-1) / (q.shape[-1] ** 0.5)
                teacher_probs = F.softmax(teacher_scores, dim=-1)
                teacher_block_probs = compress_teacher_probs(teacher_probs, args.compression_rate)
                compressed_k = build_compressed_pool(k, args.compression_rate)

            q_flat = q.flatten(1)
            k_flat = compressed_k.flatten(1)
            student_scores = indexers[layer_id](q_flat, k_flat)
            student_log_probs = F.log_softmax(student_scores, dim=-1)
            loss = F.kl_div(student_log_probs, teacher_block_probs, reduction="batchmean")
            layer_losses.append(loss)

            with torch.no_grad():
                recall_k = min(args.recall_k, teacher_block_probs.shape[-1])
                layer_recalls[str(layer_id)] = float(topk_recall(teacher_block_probs, student_scores, recall_k))
                layer_entropies[str(layer_id)] = float(
                    -(student_log_probs.exp() * student_log_probs).sum(dim=-1).mean().item()
                )

        total_loss = torch.stack(layer_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        record = {
            "step": step,
            "loss": float(total_loss.item()),
            "layer_recalls": layer_recalls,
            "layer_entropies": layer_entropies,
        }
        history.append(record)

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            recall_msg = " ".join([f"l{lid}_r@{args.recall_k}={layer_recalls[str(lid)]:.4f}" for lid in layer_ids])
            print(f"step={step} loss={record['loss']:.6f} {recall_msg} elapsed={elapsed:.1f}s")

    ckpt = {
        "layer_ids": layer_ids,
        "models": {str(layer_id): module.state_dict() for layer_id, module in indexers.items()},
        "args": vars(args),
        "history": history,
    }
    torch.save(ckpt, os.path.join(args.output_dir, "csa_multilayer_indexer_distill.pt"))

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    first = history[0]
    last = history[-1]
    summary = {
        "steps": args.steps,
        "layer_ids": layer_ids,
        "compression_rate": args.compression_rate,
        "compressed_topk": args.compressed_topk,
        "height": args.height,
        "width": args.width,
        "recall_k": args.recall_k,
        "initial_loss": first["loss"],
        "final_loss": last["loss"],
        "initial_layer_recalls": first["layer_recalls"],
        "final_layer_recalls": last["layer_recalls"],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
