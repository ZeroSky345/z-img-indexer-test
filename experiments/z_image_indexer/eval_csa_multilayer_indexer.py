import argparse
import json
import os
import time

import numpy as np
import torch

from csa_common import apply_teacher_lora, build_pipe, load_prompts, set_seed
from stage6_generate_compare_csa_single_layer import (
    csa_attention_forward,
    image_to_np,
    patchify_turbo_inputs,
    run_single_layer_with_custom_attn,
    save_comparison_grid,
)
from train_indexer_distill import BilinearIndexer
from einops import rearrange


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--model-base-path", type=str, default=None)
    parser.add_argument("--teacher-lora-path", type=str, default=None)
    parser.add_argument("--teacher-lora-alpha", type=float, default=1.0)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--indexer-ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--layer-ids", type=str, default=None)
    parser.add_argument("--compression-rate", type=int, default=2)
    parser.add_argument("--compressed-topk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_layer_ids(value: str | None, ckpt_layer_ids: list[int]) -> list[int]:
    if value is None or value.strip().lower() == "ckpt":
        return ckpt_layer_ids
    value = value.strip().lower()
    if value == "all":
        return ckpt_layer_ids
    layer_ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            layer_ids.extend(range(int(start), int(end) + 1))
        else:
            layer_ids.append(int(item))
    missing = [layer_id for layer_id in layer_ids if layer_id not in ckpt_layer_ids]
    if missing:
        raise ValueError(f"Requested layers not found in checkpoint: {missing}")
    return layer_ids


def load_multilayer_indexers(ckpt_path: str, device: str, requested_layer_ids: list[int] | None = None):
    state = torch.load(ckpt_path, map_location=device)
    ckpt_layer_ids = [int(layer_id) for layer_id in state["layer_ids"]]
    layer_ids = parse_layer_ids(None, ckpt_layer_ids) if requested_layer_ids is None else requested_layer_ids

    indexers = {}
    for layer_id in layer_ids:
        model_state = state["models"][str(layer_id)]
        q_weight = model_state["q_proj.weight"]
        input_dim = q_weight.shape[1]
        rank = q_weight.shape[0]
        indexer = BilinearIndexer(input_dim=input_dim, rank=rank).to(device)
        indexer.load_state_dict(model_state)
        indexer.eval()
        indexers[layer_id] = indexer
    return indexers, ckpt_layer_ids, state.get("args", {})


def make_multilayer_csa_model_fn(indexers, compression_rate, compressed_topk, stats_holder):
    target_layers = set(indexers.keys())

    def sparse_model_fn(
        dit,
        controlnet=None,
        latents=None,
        timestep=None,
        prompt_embeds=None,
        image_embeds=None,
        image_latents=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **kwargs,
    ):
        if dit.siglip_embedder is not None:
            raise NotImplementedError("This eval script currently supports the Z-Image-Turbo path only.")

        while isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]
        while isinstance(latents, list):
            latents = latents[0]

        unified, unified_freqs_cis, t_noisy, x_real_len, cap_real_len, patch_metadata = patchify_turbo_inputs(
            dit, latents, prompt_embeds, timestep
        )

        applied_layers = []
        for idx, layer in enumerate(dit.layers):
            if idx in target_layers:
                mod = layer.adaLN_modulation(t_noisy)
                scale_msa = 1.0 + mod.unsqueeze(1).chunk(4, dim=2)[0]
                hidden_input = layer.attention_norm1(unified) * scale_msa
                sparse_attn_out = csa_attention_forward(
                    attn_module=layer.attention,
                    hidden_states=hidden_input,
                    freqs_cis=unified_freqs_cis,
                    x_real_len=x_real_len,
                    cap_real_len=cap_real_len,
                    indexer=indexers[idx],
                    compression_rate=compression_rate,
                    compressed_topk=compressed_topk,
                )
                unified = run_single_layer_with_custom_attn(layer, unified, t_noisy, sparse_attn_out)
                applied_layers.append(idx)
            else:
                unified = layer(x=unified, attn_mask=None, freqs_cis=unified_freqs_cis, adaln_input=t_noisy)

        unified = dit.all_final_layer["2-1"](unified, t_noisy)
        x = dit.unpatchify([unified[0]], patch_metadata["x_size"])[0]
        x = rearrange(x, "C B H W -> B C H W")
        x = -x
        stats_holder.append(
            {
                "compression_rate": compression_rate,
                "compressed_topk": compressed_topk,
                "applied_layers": applied_layers,
            }
        )
        return x

    return sparse_model_fn


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    prompts = load_prompts(args.prompt_file)
    pipe = build_pipe(args)
    teacher_lora_loaded = apply_teacher_lora(pipe, args.teacher_lora_path, args.teacher_lora_alpha)
    ckpt_state = torch.load(args.indexer_ckpt, map_location=args.device)
    ckpt_layer_ids = [int(layer_id) for layer_id in ckpt_state["layer_ids"]]
    layer_ids = parse_layer_ids(args.layer_ids, ckpt_layer_ids)
    indexers, _, ckpt_args = load_multilayer_indexers(args.indexer_ckpt, args.device, layer_ids)
    original_model_fn = pipe.model_fn

    records = []
    for idx, prompt in enumerate(prompts):
        dense_start = time.perf_counter()
        dense_image = pipe(
            prompt=prompt,
            seed=args.seed + idx,
            rand_device=args.device,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
        )
        dense_time = time.perf_counter() - dense_start

        sparse_stats = []
        pipe.model_fn = make_multilayer_csa_model_fn(indexers, args.compression_rate, args.compressed_topk, sparse_stats)
        sparse_start = time.perf_counter()
        sparse_image = pipe(
            prompt=prompt,
            seed=args.seed + idx,
            rand_device=args.device,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
        )
        sparse_time = time.perf_counter() - sparse_start
        pipe.model_fn = original_model_fn

        dense_path = os.path.join(args.output_dir, f"prompt_{idx}_dense.png")
        sparse_path = os.path.join(args.output_dir, f"prompt_{idx}_csa_multilayer.png")
        grid_path = os.path.join(args.output_dir, f"prompt_{idx}_compare_csa_multilayer.png")
        dense_image.save(dense_path)
        sparse_image.save(sparse_path)
        save_comparison_grid(
            dense_image,
            sparse_image,
            "dense",
            f"CSA layers={','.join(map(str, layer_ids))} m={args.compression_rate} topk={args.compressed_topk}",
            grid_path,
        )

        dense_np = image_to_np(dense_image)
        sparse_np = image_to_np(sparse_image)
        mse = float(np.mean((dense_np - sparse_np) ** 2))
        record = {
            "prompt_index": idx,
            "prompt": prompt,
            "layer_ids": layer_ids,
            "compression_rate": args.compression_rate,
            "compressed_topk": args.compressed_topk,
            "teacher_lora_path": args.teacher_lora_path,
            "teacher_lora_alpha": args.teacher_lora_alpha,
            "teacher_lora_loaded": teacher_lora_loaded,
            "dense_time_sec": dense_time,
            "sparse_time_sec": sparse_time,
            "time_ratio_sparse_over_dense": sparse_time / dense_time if dense_time > 0 else None,
            "image_mse": mse,
            "dense_path": os.path.basename(dense_path),
            "sparse_path": os.path.basename(sparse_path),
            "compare_path": os.path.basename(grid_path),
        }
        print(json.dumps(record, ensure_ascii=False))
        records.append(record)

    summary = {
        "layer_ids": layer_ids,
        "ckpt_layer_ids": ckpt_layer_ids,
        "ckpt_args": ckpt_args,
        "compression_rate": args.compression_rate,
        "compressed_topk": args.compressed_topk,
        "teacher_lora_path": args.teacher_lora_path,
        "teacher_lora_alpha": args.teacher_lora_alpha,
        "teacher_lora_loaded": teacher_lora_loaded,
        "num_prompts": len(records),
        "mean_dense_time_sec": float(sum(r["dense_time_sec"] for r in records) / len(records)),
        "mean_sparse_time_sec": float(sum(r["sparse_time_sec"] for r in records) / len(records)),
        "mean_time_ratio_sparse_over_dense": float(
            sum(r["time_ratio_sparse_over_dense"] for r in records) / len(records)
        ),
        "mean_image_mse": float(sum(r["image_mse"] for r in records) / len(records)),
    }

    with open(os.path.join(args.output_dir, "records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
