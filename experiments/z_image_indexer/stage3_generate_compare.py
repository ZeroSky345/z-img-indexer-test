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
    compute_teacher_scores,
    project_qk,
    set_seed,
    topk_recall,
)
from diffsynth.pipelines.z_image import model_fn_z_image  # noqa: E402


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
    parser.add_argument("--topk", type=int, default=256)
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


def sparse_attention_forward(attn_module, hidden_states, freqs_cis, x_real_len, cap_real_len, indexer, topk):
    q, k = project_qk(attn_module, hidden_states, freqs_cis)
    value = attn_module.to_v(hidden_states).unflatten(-1, (attn_module.num_heads, -1))
    seq_len = hidden_states.shape[1]
    x_total_len = seq_len - cap_real_len

    with torch.amp.autocast(hidden_states.device.type, enabled=False):
        qf = q.float()
        kf = k.float()
        vf = value.float()
        dense_scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) / math.sqrt(q.shape[-1])

        image_q = qf[0, :x_real_len]
        image_k = kf[0, :x_real_len]
        student_scores = indexer(image_q.flatten(1), image_k.flatten(1))
        teacher_scores = compute_teacher_scores(image_q, image_k)
        recall = topk_recall(teacher_scores, student_scores, min(topk, x_real_len))

        topk_idx = student_scores.topk(min(topk, x_real_len), dim=-1).indices
        sparse_scores = torch.full_like(dense_scores, fill_value=torch.finfo(torch.float32).min)

        sparse_scores[:, :, x_total_len:, :] = dense_scores[:, :, x_total_len:, :]
        sparse_scores[:, :, x_real_len:x_total_len, :] = dense_scores[:, :, x_real_len:x_total_len, :]

        dense_context_positions = torch.arange(x_real_len, seq_len, device=hidden_states.device)
        for qi in range(x_real_len):
            selected_image = topk_idx[qi]
            sparse_scores[0, :, qi, dense_context_positions] = dense_scores[0, :, qi, dense_context_positions]
            sparse_scores[0, :, qi, selected_image] = dense_scores[0, :, qi, selected_image]

        probs = torch.softmax(sparse_scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", probs, vf).to(hidden_states.dtype)

    out = out.flatten(2, 3)
    out = attn_module.to_out[0](out)
    return out, float(recall)


def run_single_layer_with_custom_attn(layer, x, adaln_input, custom_attn_out):
    mod = layer.adaLN_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
    gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
    scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

    x = x + gate_msa * layer.attention_norm2(custom_attn_out)
    x = x + gate_mlp * layer.ffn_norm2(layer.feed_forward(layer.ffn_norm1(x) * scale_mlp))
    return x


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
    return unified, unified_freqs_cis, t_noisy, x_real_len, cap_real_len, patch_metadata


def make_sparse_model_fn(indexer, layer_id, topk, stats_holder):
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
            raise NotImplementedError("Stage3 script currently supports Z-Image-Turbo style path only.")

        while isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]
        while isinstance(latents, list):
            latents = latents[0]

        unified, unified_freqs_cis, t_noisy, x_real_len, cap_real_len, patch_metadata = patchify_turbo_inputs(
            dit, latents, prompt_embeds, timestep
        )

        per_step_recalls = []
        for idx, layer in enumerate(dit.layers):
            if idx == layer_id:
                mod = layer.adaLN_modulation(t_noisy)
                scale_msa = 1.0 + mod.unsqueeze(1).chunk(4, dim=2)[0]
                hidden_input = layer.attention_norm1(unified) * scale_msa
                sparse_attn_out, recall = sparse_attention_forward(
                    attn_module=layer.attention,
                    hidden_states=hidden_input,
                    freqs_cis=unified_freqs_cis,
                    x_real_len=x_real_len,
                    cap_real_len=cap_real_len,
                    indexer=indexer,
                    topk=topk,
                )
                per_step_recalls.append(recall)
                unified = run_single_layer_with_custom_attn(layer, unified, t_noisy, sparse_attn_out)
            else:
                unified = layer(x=unified, attn_mask=None, freqs_cis=unified_freqs_cis, adaln_input=t_noisy)

        unified = dit.all_final_layer["2-1"](unified, t_noisy)
        x = dit.unpatchify([unified[0]], patch_metadata["x_size"])[0]
        x = rearrange(x, "C B H W -> B C H W")
        x = -x
        stats_holder.append({"recall": per_step_recalls[0] if per_step_recalls else None})
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
        pipe.model_fn = make_sparse_model_fn(indexer, args.layer_id, args.topk, sparse_stats)
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
        sparse_path = os.path.join(args.output_dir, f"prompt_{idx}_sparse.png")
        grid_path = os.path.join(args.output_dir, f"prompt_{idx}_compare.png")
        dense_image.save(dense_path)
        sparse_image.save(sparse_path)
        save_comparison_grid(dense_image, sparse_image, "dense", f"sparse topk={args.topk}", grid_path)

        dense_np = image_to_np(dense_image)
        sparse_np = image_to_np(sparse_image)
        mse = float(np.mean((dense_np - sparse_np) ** 2))

        record = {
            "prompt_index": idx,
            "prompt": prompt,
            "topk": args.topk,
            "dense_time_sec": dense_time,
            "sparse_time_sec": sparse_time,
            "time_ratio_sparse_over_dense": sparse_time / dense_time if dense_time > 0 else None,
            "image_mse": mse,
            "mean_step_recall": float(np.mean([x["recall"] for x in sparse_stats if x["recall"] is not None])),
            "dense_path": os.path.basename(dense_path),
            "sparse_path": os.path.basename(sparse_path),
            "compare_path": os.path.basename(grid_path),
        }
        print(json.dumps(record, ensure_ascii=False))
        records.append(record)

    summary = {
        "layer_id": args.layer_id,
        "topk": args.topk,
        "num_prompts": len(records),
        "mean_dense_time_sec": float(np.mean([r["dense_time_sec"] for r in records])),
        "mean_sparse_time_sec": float(np.mean([r["sparse_time_sec"] for r in records])),
        "mean_time_ratio_sparse_over_dense": float(np.mean([r["time_ratio_sparse_over_dense"] for r in records])),
        "mean_image_mse": float(np.mean([r["image_mse"] for r in records])),
        "mean_step_recall": float(np.mean([r["mean_step_recall"] for r in records])),
    }

    with open(os.path.join(args.output_dir, "records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
