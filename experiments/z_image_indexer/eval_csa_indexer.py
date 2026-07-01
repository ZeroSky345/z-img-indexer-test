import argparse
import json
import os

from csa_common import apply_teacher_lora, build_pipe, load_prompts, set_seed
from stage6_generate_compare_csa_single_layer import load_indexer, make_csa_model_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--model-base-path", type=str, default=None)
    parser.add_argument("--teacher-lora-path", type=str, default=None)
    parser.add_argument("--teacher-lora-alpha", type=float, default=1.0)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--indexer-ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--layer-id", type=int, default=12)
    parser.add_argument("--compression-rate", type=int, default=2)
    parser.add_argument("--compressed-topk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    prompts = load_prompts(args.prompt_file)
    pipe = build_pipe(args)
    teacher_lora_loaded = apply_teacher_lora(pipe, args.teacher_lora_path, args.teacher_lora_alpha)
    indexer = load_indexer(args.indexer_ckpt, args.device)
    original_model_fn = pipe.model_fn

    from stage6_generate_compare_csa_single_layer import image_to_np, save_comparison_grid
    import numpy as np
    import time

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
            f"CSA eval layer={args.layer_id} m={args.compression_rate} topk={args.compressed_topk}",
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
        "layer_id": args.layer_id,
        "compression_rate": args.compression_rate,
        "compressed_topk": args.compressed_topk,
        "teacher_lora_path": args.teacher_lora_path,
        "teacher_lora_alpha": args.teacher_lora_alpha,
        "teacher_lora_loaded": teacher_lora_loaded,
        "num_prompts": len(records),
        "mean_dense_time_sec": float(sum(r["dense_time_sec"] for r in records) / len(records)),
        "mean_sparse_time_sec": float(sum(r["sparse_time_sec"] for r in records) / len(records)),
        "mean_time_ratio_sparse_over_dense": float(sum(r["time_ratio_sparse_over_dense"] for r in records) / len(records)),
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
