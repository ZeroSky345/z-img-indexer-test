# Z-Image Indexer 训练服务器启动说明

本文档说明如何在训练服务器上直接启动当前项目的 `Z-Image-Turbo + LoRA adapter 权重 + CSA/indexer` 训练与评测。

公开仓库：

- https://github.com/ZeroSky345/z-img-indexer-test

## 1. 当前训练内容

本项目当前不做 `Z-Image-Turbo` 主模型全量训练；LoRA 部分的含义是训练 LoRA adapter 权重。

注意：`--lora_base_model dit` 表示 LoRA adapter 挂到 DiT 上，base DiT 原始参数冻结，只更新 LoRA adapter 参数，不是全量训练主模型 DiT。

当前训练对象分两类：

- LoRA adapter 权重
- CSA-like sparse routing indexer
- 单层 indexer：`layer12`
- 全层 indexer：`layer_ids=all`

LoRA 的角色是：

- 先通过 LoRA adapter 权重训练得到 `.safetensors`
- 在 indexer 蒸馏时加载到 frozen teacher DiT
- 让 indexer 学习 `Z-Image-Turbo + LoRA` 后的 attention/QK 分布

也就是说，训练目标是：

```text
frozen Z-Image-Turbo + trainable LoRA adapter -> LoRA weight
frozen Z-Image-Turbo + selected LoRA weight -> teacher
trainable CSA/indexer -> student
```

## 2. 文件入口

核心训练脚本：

- `experiments/z_image_indexer/train_csa_indexer.py`
- `experiments/z_image_indexer/train_csa_multilayer_indexer.py`

核心评测脚本：

- `experiments/z_image_indexer/eval_csa_indexer.py`
- `experiments/z_image_indexer/eval_csa_multilayer_indexer.py`

推荐启动脚本：

- `experiments/z_image_indexer/run_train_lora_weights_2048_adapter.sh`
- `experiments/z_image_indexer/run_train_lora_weights_2048.sh`
- `experiments/z_image_indexer/run_train_csa_m2_topk64_2048_lora_teacher.sh`
- `experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh`
- `experiments/z_image_indexer/run_eval_csa_m2_topk64_2048_lora_teacher.sh`
- `experiments/z_image_indexer/run_eval_csa_all_layers_m2_topk64_2048_lora_teacher.sh`

## 3. 服务器准备

克隆仓库：

```bash
git clone https://github.com/ZeroSky345/z-img-indexer-test.git
cd z-img-indexer-test
```

准备 Python 环境。若服务器已有可运行 DiffSynth/Z-Image 的环境，可直接使用。

基本依赖：

- `torch`
- `transformers`
- `modelscope`
- `safetensors`
- `einops`
- `Pillow`

快速检查：

```bash
python - <<'PY'
import torch
import diffsynth
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("diffsynth import ok")
PY
```

## 4. 模型与 LoRA 权重路径

默认基础模型路径：

```bash
/tmp/DiffSynth-Studio/models
```

如果模型在其他位置，修改启动脚本里的：

```bash
--model-base-path /tmp/DiffSynth-Studio/models
```

LoRA 权重默认路径：

```bash
./models/lora/z_image_turbo_lora.safetensors
```

如果已经有 LoRA 权重，直接放置：

```bash
mkdir -p ./models/lora
cp /path/to/your_lora.safetensors ./models/lora/z_image_turbo_lora.safetensors
```

也可以不复制，直接用环境变量覆盖：

```bash
export TEACHER_LORA_PATH=/path/to/your_lora.safetensors
```

LoRA alpha 默认：

```bash
export TEACHER_LORA_ALPHA=1.0
```

## 5. LoRA adapter 权重训练

推荐先训练 LoRA adapter 权重：

```bash
bash experiments/z_image_indexer/run_train_lora_weights_2048_adapter.sh
```

这个脚本会：

- 下载示例数据 `DiffSynth-Studio/diffsynth_example_dataset`
- 下载 `ostris/zimage_turbo_training_adapter`
- 以 2048x2048 配置训练 LoRA adapter 权重
- 冻结 Z-Image-Turbo base DiT，只更新 LoRA adapter 参数

默认输出目录：

```text
./models/lora/z_image_turbo_lora_2048_adapter/
```

训练后选择一个 checkpoint 作为 indexer teacher LoRA，例如：

```bash
mkdir -p ./models/lora
cp ./models/lora/z_image_turbo_lora_2048_adapter/epoch-4.safetensors \
  ./models/lora/z_image_turbo_lora.safetensors
```

也可以不复制，直接设置：

```bash
export TEACHER_LORA_PATH=./models/lora/z_image_turbo_lora_2048_adapter/epoch-4.safetensors
```

如果不想使用 preset adapter，可以运行普通 LoRA 训练入口：

```bash
bash experiments/z_image_indexer/run_train_lora_weights_2048.sh
```

## 6. 环境变量

建议：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODELSCOPE_CACHE=/tmp/modelscope-cache
export HF_HOME=/tmp/hf-home
export TEACHER_LORA_PATH=./models/lora/z_image_turbo_lora.safetensors
export TEACHER_LORA_ALPHA=1.0
```

注意：当前 indexer 训练脚本仍是单进程入口；`CUDA_VISIBLE_DEVICES=0,1,2,3` 表示面向多卡服务器环境，但不等于已经实现 DDP/model parallel。

## 7. 单层 indexer 训练

推荐先跑单层，确认 LoRA teacher、数据 prompt、模型路径全部正常。

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64_2048_lora_teacher.sh
```

等价核心参数：

```bash
python experiments/z_image_indexer/train_csa_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
  --teacher-lora-path ./models/lora/z_image_turbo_lora.safetensors \
  --teacher-lora-alpha 1.0 \
  --prompt-file experiments/z_image_indexer/configs/default_prompts_train.txt \
  --output-dir experiments/z_image_indexer/results/train_csa_layer12_m2_topk64_2048_lora_teacher \
  --steps 1000 \
  --height 2048 \
  --width 2048 \
  --num-inference-steps 4 \
  --layer-id 12 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --rank 128 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --recall-k 16 \
  --query-chunk-size 512 \
  --metrics-max-queries 2048 \
  --device cuda
```

输出目录：

```text
experiments/z_image_indexer/results/train_csa_layer12_m2_topk64_2048_lora_teacher/
```

关键输出：

- `csa_indexer_distill.pt`
- `metrics.json`
- `summary.json`
- `run_config.json`

## 8. 全层 indexer 训练

单层确认无误后，启动全层训练：

```bash
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh
```

等价核心参数：

```bash
python experiments/z_image_indexer/train_csa_multilayer_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
  --teacher-lora-path ./models/lora/z_image_turbo_lora.safetensors \
  --teacher-lora-alpha 1.0 \
  --prompt-file experiments/z_image_indexer/configs/default_prompts_train.txt \
  --output-dir experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4 \
  --steps 1000 \
  --height 2048 \
  --width 2048 \
  --num-inference-steps 4 \
  --layer-ids all \
  --batch-size 4 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --rank 128 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --recall-k 16 \
  --query-chunk-size 512 \
  --metrics-max-queries 2048 \
  --log-every 10 \
  --device cuda
```

输出目录：

```text
experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4/
```

关键输出：

- `csa_multilayer_indexer_distill.pt`
- `metrics.json`
- `summary.json`
- `run_config.json`

## 9. 单层评测

训练完单层 indexer 后执行：

```bash
bash experiments/z_image_indexer/run_eval_csa_m2_topk64_2048_lora_teacher.sh
```

默认 checkpoint：

```text
experiments/z_image_indexer/results/train_csa_layer12_m2_topk64_2048_lora_teacher/csa_indexer_distill.pt
```

如需指定：

```bash
INDEXER_CKPT=/path/to/csa_indexer_distill.pt \
bash experiments/z_image_indexer/run_eval_csa_m2_topk64_2048_lora_teacher.sh
```

输出目录：

```text
experiments/z_image_indexer/results/eval_csa_layer12_m2_topk64_2048_lora_teacher/
```

## 10. 全层评测

训练完全层 indexer 后执行：

```bash
bash experiments/z_image_indexer/run_eval_csa_all_layers_m2_topk64_2048_lora_teacher.sh
```

默认 checkpoint：

```text
experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4/csa_multilayer_indexer_distill.pt
```

如需指定：

```bash
INDEXER_CKPT=/path/to/csa_multilayer_indexer_distill.pt \
bash experiments/z_image_indexer/run_eval_csa_all_layers_m2_topk64_2048_lora_teacher.sh
```

输出目录：

```text
experiments/z_image_indexer/results/eval_csa_all_layers_m2_topk64_2048_lora_teacher/
```

评测输出：

- `records.json`
- `summary.json`
- `run_config.json`
- `prompt_*_dense.png`
- `prompt_*_csa_multilayer.png`
- `prompt_*_compare_csa_multilayer.png`

## 11. 指标查看

训练后先看：

```bash
cat experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4/summary.json
```

重点字段：

- `initial_loss`
- `final_loss`
- `initial_layer_recalls`
- `final_layer_recalls`
- `teacher_lora_loaded`
- `teacher_lora_path`

评测后看：

```bash
cat experiments/z_image_indexer/results/eval_csa_all_layers_m2_topk64_2048_lora_teacher/summary.json
```

重点字段：

- `mean_dense_time_sec`
- `mean_sparse_time_sec`
- `mean_time_ratio_sparse_over_dense`
- `mean_image_mse`
- `teacher_lora_loaded`

## 12. 推荐执行顺序

1. 训练 LoRA adapter 权重，推荐运行 `run_train_lora_weights_2048_adapter.sh`。
2. 选择一个 LoRA checkpoint，放到 `./models/lora/z_image_turbo_lora.safetensors`，或设置 `TEACHER_LORA_PATH`。
3. 跑单层 LoRA-teacher indexer 训练。
4. 跑单层 LoRA-teacher indexer 评测。
5. 跑全层 LoRA-teacher indexer 训练。
6. 跑全层 LoRA-teacher indexer 评测。
7. 汇总 `summary.json`、`metrics.json` 和对比图。

## 13. 给执行者的最短说明

```text
请 clone https://github.com/ZeroSky345/z-img-indexer-test，
进入仓库后准备好 Python / CUDA / DiffSynth 环境。

先训练 LoRA adapter 权重：
bash experiments/z_image_indexer/run_train_lora_weights_2048_adapter.sh

选择输出 checkpoint，例如：
./models/lora/z_image_turbo_lora_2048_adapter/epoch-4.safetensors

复制为 indexer teacher LoRA：
mkdir -p ./models/lora
cp ./models/lora/z_image_turbo_lora_2048_adapter/epoch-4.safetensors \
  ./models/lora/z_image_turbo_lora.safetensors

如果已经有训练好的 LoRA 权重，也可以直接放到：
./models/lora/z_image_turbo_lora.safetensors

然后依次训练和评测 indexer：
bash experiments/z_image_indexer/run_train_csa_m2_topk64_2048_lora_teacher.sh
bash experiments/z_image_indexer/run_eval_csa_m2_topk64_2048_lora_teacher.sh
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_2048_lora_teacher_bs4.sh
bash experiments/z_image_indexer/run_eval_csa_all_layers_m2_topk64_2048_lora_teacher.sh

最后返回 train/eval 目录下的 summary.json、metrics.json、run_config.json 和对比图。
```
