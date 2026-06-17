import argparse
import json
import math
import os
import types
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diffsynth.pipelines.z_image import model_fn_z_image_turbo

from train_indexer_distill import (
    DEFAULT_PROMPTS,
    BilinearIndexer,
    build_noise_latents,
    build_pipe,
    capture_prompt_embedder,
    compute_teacher_scores,
    patchify_turbo_inputs,
    project_qk,
    set_seed,
    topk_recall,
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
    parser.add_argument("--topk-list", type=str, default="32,64,128,256")
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


def dense_attention_output(attn_module, hidden_states, freqs_cis):
    q, k = project_qk(attn_module, hidden_states, freqs_cis)
    value = attn_module.to_v(hidden_states).unflatten(-1, (attn_module.num_heads, -1))
    with torch.amp.autocast(hidden_states.device.type, enabled=False):
        qf = q.float()
        kf = k.float()
        vf = value.float()
        scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) / math.sqrt(q.shape[-1])
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", probs, vf).to(hidden_states.dtype)
    out = out.flatten(2, 3)
    out = attn_module.to_out[0](out)
    return out, q.detach(), k.detach(), value.detach()


def sparse_attention_output(
    attn_module,
    hidden_states,
    freqs_cis,
    indexer,
    x_real_len: int,
    x_total_len: int,
    topk: int,
):
    dense_out, q, k, value = dense_attention_output(attn_module, hidden_states, freqs_cis)

    device = hidden_states.device
    seq_len = hidden_states.shape[1]
    image_q = q[0, :x_real_len].float()
    image_k = k[0, :x_real_len].float()
    q_flat = image_q.flatten(1)
    k_flat = image_k.flatten(1)
    student_scores = indexer(q_flat, k_flat)
    teacher_scores = compute_teacher_scores(image_q, image_k)
    recall = topk_recall(teacher_scores, student_scores, min(topk, x_real_len))

    topk_idx = student_scores.topk(min(topk, x_real_len), dim=-1).indices
    sparse_scores = torch.full(
        (1, attn_module.num_heads, seq_len, seq_len),
        fill_value=torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=device,
    )

    with torch.amp.autocast(hidden_states.device.type, enabled=False):
        dense_scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(q.shape[-1])

    # Keep all non-image queries dense.
    sparse_scores[:, :, x_total_len:, :] = dense_scores[:, :, x_total_len:, :]
    sparse_scores[:, :, x_real_len:x_total_len, :] = dense_scores[:, :, x_real_len:x_total_len, :]

    dense_context_positions = torch.arange(x_real_len, seq_len, device=device)
    for qi in range(x_real_len):
        selected_image = topk_idx[qi]
        sparse_scores[0, :, qi, dense_context_positions] = dense_scores[0, :, qi, dense_context_positions]
        sparse_scores[0, :, qi, selected_image] = dense_scores[0, :, qi, selected_image]

    with torch.amp.autocast(hidden_states.device.type, enabled=False):
        probs = torch.softmax(sparse_scores, dim=-1)
        sparse_out = torch.einsum("bhqk,bkhd->bqhd", probs, value.float()).to(hidden_states.dtype)
    sparse_out = sparse_out.flatten(2, 3)
    sparse_out = attn_module.to_out[0](sparse_out)

    return dense_out, sparse_out, recall


def run_single_layer_with_custom_attn(layer, x, freqs_cis, adaln_input, custom_attn_out):
    mod = layer.adaLN_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
    gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
    scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

    x = x + gate_msa * layer.attention_norm2(custom_attn_out)
    x = x + gate_mlp * layer.ffn_norm2(layer.feed_forward(layer.ffn_norm1(x) * scale_mlp))
    return x


def forward_turbo_sparse(pipe, prompt_embeds, latents, timestep, layer_id, indexer, topk):
    dit = pipe.dit
    timestep_for_fn = timestep
    baseline = model_fn_z_image_turbo(
        dit=dit,
        latents=latents,
        timestep=timestep_for_fn,
        prompt_embeds=prompt_embeds,
        image_embeds=None,
        image_latents=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    )

    unified, unified_freqs_cis, t_noisy, x_real_len, cap_real_len = patchify_turbo_inputs(
        dit, latents, prompt_embeds, timestep_for_fn
    )
    x_total_len = unified.shape[1] - cap_real_len

    layer_metrics = {}
    for idx, layer in enumerate(dit.layers):
        if idx == layer_id:
            hidden_input = layer.attention_norm1(unified) * (1.0 + layer.adaLN_modulation(t_noisy).unsqueeze(1).chunk(4, dim=2)[0])
            dense_attn_out, sparse_attn_out, recall = sparse_attention_output(
                layer.attention,
                hidden_input,
                unified_freqs_cis,
                indexer,
                x_real_len=x_real_len,
                x_total_len=x_total_len,
                topk=topk,
            )
            layer_metrics["recall_at_k"] = float(recall)
            layer_metrics["attn_output_mse"] = float(F.mse_loss(sparse_attn_out[:, :x_real_len], dense_attn_out[:, :x_real_len]).item())
            layer_metrics["attn_output_cosine"] = float(
                F.cosine_similarity(
                    sparse_attn_out[:, :x_real_len].reshape(1, -1),
                    dense_attn_out[:, :x_real_len].reshape(1, -1),
                    dim=-1,
                ).item()
            )
            unified = run_single_layer_with_custom_attn(layer, unified, unified_freqs_cis, t_noisy, sparse_attn_out)
        else:
            unified = layer(x=unified, attn_mask=None, freqs_cis=unified_freqs_cis, adaln_input=t_noisy)

    unified = dit.all_final_layer["2-1"](unified, t_noisy)
    sparse = dit.unpatchify([unified[0]], [(1, latents.shape[2], latents.shape[3])])[0]
    sparse = rearrange(sparse, "C B H W -> B C H W")
    sparse = -sparse

    final_mse = float(F.mse_loss(sparse, baseline).item())
    final_cos = float(F.cosine_similarity(sparse.reshape(1, -1), baseline.reshape(1, -1), dim=-1).item())
    layer_metrics["final_noise_pred_mse"] = final_mse
    layer_metrics["final_noise_pred_cosine"] = final_cos
    return layer_metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    topk_list = [int(x) for x in args.topk_list.split(",") if x.strip()]
    pipe = build_pipe(args)
    indexer = load_indexer(args.indexer_ckpt, args.device)
    prompt_cache = {prompt: capture_prompt_embedder(pipe, prompt) for prompt in DEFAULT_PROMPTS}

    timestep_values = [t.unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype) for t in pipe.scheduler.timesteps]
    results = []

    for prompt_idx, prompt in enumerate(DEFAULT_PROMPTS):
        prompt_embeds = prompt_cache[prompt]
        for step_idx, timestep in enumerate(timestep_values):
            latents = build_noise_latents(pipe, args.height, args.width, args.seed + prompt_idx * 100 + step_idx)
            for topk in topk_list:
                metrics = forward_turbo_sparse(
                    pipe=pipe,
                    prompt_embeds=prompt_embeds,
                    latents=latents,
                    timestep=timestep,
                    layer_id=args.layer_id,
                    indexer=indexer,
                    topk=topk,
                )
                record = {
                    "prompt_index": prompt_idx,
                    "step_index": step_idx,
                    "topk": topk,
                    **metrics,
                }
                print(json.dumps(record, ensure_ascii=False))
                results.append(record)

    summary = {}
    for topk in topk_list:
        subset = [r for r in results if r["topk"] == topk]
        summary[str(topk)] = {
            "mean_recall_at_k": float(sum(r["recall_at_k"] for r in subset) / len(subset)),
            "mean_attn_output_mse": float(sum(r["attn_output_mse"] for r in subset) / len(subset)),
            "mean_attn_output_cosine": float(sum(r["attn_output_cosine"] for r in subset) / len(subset)),
            "mean_final_noise_pred_mse": float(sum(r["final_noise_pred_mse"] for r in subset) / len(subset)),
            "mean_final_noise_pred_cosine": float(sum(r["final_noise_pred_cosine"] for r in subset) / len(subset)),
        }

    with open(os.path.join(args.output_dir, "records.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
