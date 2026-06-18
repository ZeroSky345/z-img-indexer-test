import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from einops import rearrange

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_indexer_distill import (  # noqa: E402
    DEFAULT_PROMPTS,
    BilinearIndexer,
    build_pipe,
    project_qk,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--model-base-path", type=str, default=None)
    parser.add_argument("--indexer-ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--layer-id", type=int, default=12)
    parser.add_argument("--compression-rate", type=int, default=4)
    parser.add_argument("--compressed-topk", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_indexer(ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    model_state = state["model_state_dict"]
    q_weight = model_state["q_proj.weight"]
    input_dim = q_weight.shape[1]
    rank = q_weight.shape[0]
    indexer = BilinearIndexer(input_dim=input_dim, rank=rank).to(device)
    indexer.load_state_dict(model_state)
    indexer.eval()
    return indexer


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
    if x.dim() == 2:
        x = rearrange(x, "L C -> 1 L C")
    if x_freqs_cis.dim() == 2:
        x_freqs_cis = rearrange(x_freqs_cis, "L C -> 1 L C")

    for layer in dit.noise_refiner:
        x = layer(x=x, attn_mask=None, freqs_cis=x_freqs_cis, adaln_input=t_noisy)

    cap_feats = dit.cap_embedder(cap_feats)
    cap_feats[patch_metadata["cap_pad_mask"][0]] = dit.cap_pad_token.to(dtype=x.dtype, device=x.device)
    cap_freqs_cis = dit.rope_embedder(torch.cat(patch_metadata["cap_pos_ids"], dim=0))
    if cap_feats.dim() == 2:
        cap_feats = rearrange(cap_feats, "L C -> 1 L C")
    if cap_freqs_cis.dim() == 2:
        cap_freqs_cis = rearrange(cap_freqs_cis, "L C -> 1 L C")

    for layer in dit.context_refiner:
        cap_feats = layer(x=cap_feats, attn_mask=None, freqs_cis=cap_freqs_cis)

    unified = torch.cat([x, cap_feats], dim=1)
    unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)
    return unified, unified_freqs_cis, t_noisy, x_real_len, cap_real_len, patch_metadata


def build_compressed_pool(image_k, image_v, compression_rate):
    num_tokens, num_heads, head_dim = image_k.shape
    block_count = math.ceil(num_tokens / compression_rate)
    pad = block_count * compression_rate - num_tokens
    if pad > 0:
        image_k = torch.cat([image_k, image_k[-1:].expand(pad, -1, -1)], dim=0)
        image_v = torch.cat([image_v, image_v[-1:].expand(pad, -1, -1)], dim=0)
    image_k = image_k.view(block_count, compression_rate, num_heads, head_dim).mean(dim=1)
    image_v = image_v.view(block_count, compression_rate, num_heads, head_dim).mean(dim=1)
    return image_k, image_v


def csa_attention_forward(attn_module, hidden_states, freqs_cis, x_real_len, cap_real_len, indexer, compression_rate, compressed_topk):
    q, k = project_qk(attn_module, hidden_states, freqs_cis)
    value = attn_module.to_v(hidden_states).unflatten(-1, (attn_module.num_heads, -1))
    seq_len = hidden_states.shape[1]

    with torch.amp.autocast(hidden_states.device.type, enabled=False):
        qf = q.float()[0]
        kf = k.float()[0]
        vf = value.float()[0]

        image_q = qf[:x_real_len]
        image_k = kf[:x_real_len]
        image_v = vf[:x_real_len]
        context_k = kf[x_real_len:]
        context_v = vf[x_real_len:]

        comp_k, comp_v = build_compressed_pool(image_k, image_v, compression_rate)
        student_scores = indexer(image_q.flatten(1), comp_k.flatten(1))
        topk = min(compressed_topk, comp_k.shape[0])
        topk_idx = student_scores.topk(topk, dim=-1).indices

        gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, comp_k.shape[1], comp_k.shape[2])
        selected_k = comp_k.unsqueeze(0).expand(x_real_len, -1, -1, -1).gather(1, gather_idx)
        selected_v = comp_v.unsqueeze(0).expand(x_real_len, -1, -1, -1).gather(1, gather_idx)

        if context_k.shape[0] > 0:
            context_k_exp = context_k.unsqueeze(0).expand(x_real_len, -1, -1, -1)
            context_v_exp = context_v.unsqueeze(0).expand(x_real_len, -1, -1, -1)
            candidate_k = torch.cat([selected_k, context_k_exp], dim=1)
            candidate_v = torch.cat([selected_v, context_v_exp], dim=1)
        else:
            candidate_k = selected_k
            candidate_v = selected_v

        candidate_scores = torch.einsum("qhd,qchd->qhc", image_q, candidate_k) / math.sqrt(image_q.shape[-1])
        candidate_probs = torch.softmax(candidate_scores, dim=-1)
        sparse_image_out = torch.einsum("qhc,qchd->qhd", candidate_probs, candidate_v)

        dense_scores_non_image = torch.einsum("qhd,khd->hqk", qf[x_real_len:], kf) / math.sqrt(qf.shape[-1]) if x_real_len < seq_len else None
        if dense_scores_non_image is not None:
            non_image_probs = torch.softmax(dense_scores_non_image, dim=-1)
            non_image_out = torch.einsum("hqk,khd->qhd", non_image_probs, vf)
            full_out = torch.cat([sparse_image_out, non_image_out], dim=0)
        else:
            full_out = sparse_image_out

    out = full_out.unsqueeze(0).flatten(2, 3).to(hidden_states.dtype)
    out = attn_module.to_out[0](out)
    return out


def run_single_layer_with_custom_attn(layer, x, adaln_input, custom_attn_out):
    mod = layer.adaLN_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
    gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
    scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
    x = x + gate_msa * layer.attention_norm2(custom_attn_out)
    x = x + gate_mlp * layer.ffn_norm2(layer.feed_forward(layer.ffn_norm1(x) * scale_mlp))
    return x


def make_csa_model_fn(indexer, layer_id, compression_rate, compressed_topk, stats_holder):
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
            raise NotImplementedError("Stage6 script currently supports Z-Image-Turbo style path only.")

        while isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]
        while isinstance(latents, list):
            latents = latents[0]

        unified, unified_freqs_cis, t_noisy, x_real_len, cap_real_len, patch_metadata = patchify_turbo_inputs(
            dit, latents, prompt_embeds, timestep
        )

        for idx, layer in enumerate(dit.layers):
            if idx == layer_id:
                mod = layer.adaLN_modulation(t_noisy)
                scale_msa = 1.0 + mod.unsqueeze(1).chunk(4, dim=2)[0]
                hidden_input = layer.attention_norm1(unified) * scale_msa
                sparse_attn_out = csa_attention_forward(
                    attn_module=layer.attention,
                    hidden_states=hidden_input,
                    freqs_cis=unified_freqs_cis,
                    x_real_len=x_real_len,
                    cap_real_len=cap_real_len,
                    indexer=indexer,
                    compression_rate=compression_rate,
                    compressed_topk=compressed_topk,
                )
                unified = run_single_layer_with_custom_attn(layer, unified, t_noisy, sparse_attn_out)
            else:
                unified = layer(x=unified, attn_mask=None, freqs_cis=unified_freqs_cis, adaln_input=t_noisy)

        unified = dit.all_final_layer["2-1"](unified, t_noisy)
        x = dit.unpatchify([unified[0]], patch_metadata["x_size"])[0]
        x = rearrange(x, "C B H W -> B C H W")
        x = -x
        stats_holder.append({"compression_rate": compression_rate, "compressed_topk": compressed_topk})
        return x

    return sparse_model_fn


def image_to_np(image: Image.Image):
    return np.asarray(image).astype(np.float32) / 255.0


def save_comparison_grid(dense: Image.Image, sparse: Image.Image, title_dense: str, title_sparse: str, out_path: str):
    width, height = dense.size
    canvas = Image.new("RGB", (width * 2, height + 40), "white")
    canvas.paste(dense, (0, 40))
    canvas.paste(sparse, (width, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title_dense, fill="black")
    draw.text((width + 10, 10), title_sparse, fill="black")
    canvas.save(out_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    prompts = DEFAULT_PROMPTS[: args.num_prompts]
    pipe = build_pipe(args)
    indexer = load_indexer(args.indexer_ckpt, args.device)
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
        pipe.model_fn = make_csa_model_fn(indexer, args.layer_id, args.compression_rate, args.compressed_topk, sparse_stats)
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
        sparse_path = os.path.join(args.output_dir, f"prompt_{idx}_csa.png")
        grid_path = os.path.join(args.output_dir, f"prompt_{idx}_compare_csa.png")
        dense_image.save(dense_path)
        sparse_image.save(sparse_path)
        save_comparison_grid(
            dense_image,
            sparse_image,
            "dense",
            f"CSA-like layer={args.layer_id} m={args.compression_rate} topk={args.compressed_topk}",
            grid_path,
        )

        dense_np = image_to_np(dense_image)
        sparse_np = image_to_np(sparse_image)
        mse = float(np.mean((dense_np - sparse_np) ** 2))

        record = {
            "prompt_index": idx,
            "prompt": prompt,
            "layer_id": args.layer_id,
            "compression_rate": args.compression_rate,
            "compressed_topk": args.compressed_topk,
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
        "layer_id": args.layer_id,
        "compression_rate": args.compression_rate,
        "compressed_topk": args.compressed_topk,
        "num_prompts": len(records),
        "mean_dense_time_sec": float(np.mean([r["dense_time_sec"] for r in records])),
        "mean_sparse_time_sec": float(np.mean([r["sparse_time_sec"] for r in records])),
        "mean_time_ratio_sparse_over_dense": float(np.mean([r["time_ratio_sparse_over_dense"] for r in records])),
        "mean_image_mse": float(np.mean([r["image_mse"] for r in records])),
    }

    with open(os.path.join(args.output_dir, "records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
