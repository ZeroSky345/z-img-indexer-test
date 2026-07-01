import argparse
import json
import os
import random
import time

import torch

from csa_common import (
    BilinearIndexer,
    apply_teacher_lora,
    build_compressed_pool,
    build_noise_latents,
    build_pipe,
    cache_prompt_embeds,
    capture_teacher_qk,
    csa_distill_loss_metrics,
    load_prompts,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--model-base-path", type=str, default=None)
    parser.add_argument("--teacher-lora-path", type=str, default=None)
    parser.add_argument("--teacher-lora-alpha", type=float, default=1.0)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--layer-id", type=int, default=12)
    parser.add_argument("--compression-rate", type=int, default=2)
    parser.add_argument("--compressed-topk", type=int, default=64)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--recall-k", type=int, default=16)
    parser.add_argument("--query-chunk-size", type=int, default=0)
    parser.add_argument("--metrics-max-queries", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    pipe = build_pipe(args)
    teacher_lora_loaded = apply_teacher_lora(pipe, args.teacher_lora_path, args.teacher_lora_alpha)
    prompts = load_prompts(args.prompt_file)
    prompt_cache = cache_prompt_embeds(pipe, prompts)

    sample_q, sample_k, x_real_len, _ = capture_teacher_qk(
        pipe=pipe,
        prompt_embeds=prompt_cache[prompts[0]],
        latents=build_noise_latents(pipe, args.height, args.width, args.seed),
        timestep=pipe.scheduler.timesteps[0].unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype),
        layer_id=args.layer_id,
    )
    sample_q = sample_q[0, :x_real_len].float()
    sample_k = sample_k[0, :x_real_len].float()
    compressed_k = build_compressed_pool(sample_k, args.compression_rate)
    input_dim = compressed_k.flatten(1).shape[-1]

    indexer = BilinearIndexer(input_dim=input_dim, rank=args.rank).to(pipe.device)
    optimizer = torch.optim.AdamW(indexer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    start = time.time()
    for step in range(1, args.steps + 1):
        prompt = random.choice(prompts)
        prompt_embeds = prompt_cache[prompt]
        timestep = random.choice(pipe.scheduler.timesteps).unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype)
        latents = build_noise_latents(pipe, args.height, args.width, args.seed + step)

        with torch.no_grad():
            q, k, x_real_len, _ = capture_teacher_qk(pipe, prompt_embeds, latents, timestep, args.layer_id)
            q = q[0, :x_real_len].float()
            k = k[0, :x_real_len].float()

        loss, recall, entropy = csa_distill_loss_metrics(
            indexer=indexer,
            image_q=q,
            image_k=k,
            compression_rate=args.compression_rate,
            recall_k=args.recall_k,
            query_chunk_size=args.query_chunk_size,
            metrics_max_queries=args.metrics_max_queries,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

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
        os.path.join(args.output_dir, "csa_indexer_distill.pt"),
    )
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    summary = {
        "steps": args.steps,
        "layer_id": args.layer_id,
        "compression_rate": args.compression_rate,
        "compressed_topk": args.compressed_topk,
        "height": args.height,
        "width": args.width,
        "recall_k": args.recall_k,
        "teacher_lora_path": args.teacher_lora_path,
        "teacher_lora_alpha": args.teacher_lora_alpha,
        "teacher_lora_loaded": teacher_lora_loaded,
        "query_chunk_size": args.query_chunk_size,
        "metrics_max_queries": args.metrics_max_queries,
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
