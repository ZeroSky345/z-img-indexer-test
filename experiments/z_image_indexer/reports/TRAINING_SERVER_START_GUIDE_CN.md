# Z-Image Indexer 训练服务器启动说明

本文档用于把公开仓库搬到新的训练机器后，直接启动当前的 Z-Image indexer 训练。

公开仓库：

- https://github.com/ZeroSky345/z-img-indexer-test

当前训练对象不是重训完整 `Z-Image-Turbo`，而是训练一个独立的 sparse routing / indexer 模块。

## 1. 当前可以启动的训练任务

### 任务 A：单层 CSA indexer 训练

这是当前最稳的默认入口，适合先确认环境、模型路径和训练脚本是否可用。

配置：

- 模型：`Tongyi-MAI/Z-Image-Turbo`
- 层：`layer12`
- 路线：`CSA-like compressed memory`
- `compression_rate=2`
- `compressed_topk=64`
- 分辨率：`512x512`
- 推理步数：`4`
- 训练模式：只训练 indexer，不训练主模型

启动命令：

```bash
bash experiments/z_image_indexer/run_train_csa_m2_topk64.sh
```

### 任务 B：双层 CSA indexer 训练

这是此前已经验证过可以跑通的多层入口，用于训练 `layer12,13` 两层 indexer。

启动命令：

```bash
bash experiments/z_image_indexer/run_train_csa_multilayer_l12_l13_m2_topk64.sh
```

### 任务 C：全部主干层 CSA indexer 训练

这是最新补充的入口，支持训练 Z-Image DiT 的全部 main transformer layers。

配置：

- 层：`all`
- 默认等价于 Z-Image DiT 所有主干层，当前模型通常是 `0-29`
- `batch_size=4`
- `compression_rate=2`
- `compressed_topk=64`
- 训练模式：只训练 indexer，不训练主模型

启动命令：

```bash
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_bs4.sh
```

说明：

- `--layer-ids all` 会自动读取 `pipe.dit.layers` 的层数。
- `--batch-size 4` 是梯度累积意义上的 batch size，每个 optimizer step 累计 4 个 prompt/timestep/latent 样本。
- 全层训练比单层和双层明显更重，建议先做短步数 smoke run。

## 2. 推荐硬件

### 单层训练

建议：

- NVIDIA GPU
- 能正常运行 `Z-Image-Turbo` 512x512 4-step 推理
- 推荐 40GB 以上显存，更稳妥是 80GB

### 双层训练

建议：

- 单张 80GB GPU 优先
- 如果环境已经能跑单层训练，双层训练通常可以继续尝试

### 全层训练

建议：

- 单张 80GB GPU 先做 smoke run
- 默认 `batch_size=4`
- 如果 OOM，按顺序降级：

```bash
--batch-size 2
```

再不行：

```bash
--batch-size 1
```

如果全层 `batch_size=1` 仍然 OOM，才考虑多卡或更细的层分组训练。

## 3. 服务器环境准备

### 3.1 克隆仓库

```bash
git clone https://github.com/ZeroSky345/z-img-indexer-test.git
cd z-img-indexer-test
```

### 3.2 准备 Python 环境

如果服务器已有能运行 `DiffSynth-Studio / Z-Image-Turbo` 的环境，可以直接使用已有环境。

如果需要新建环境，可参考：

```bash
python -m venv /tmp/diffsynth-venv
source /tmp/diffsynth-venv/bin/activate
python -m pip install --upgrade pip
```

然后安装项目依赖。不同训练服务器 CUDA / PyTorch 版本可能不同，建议优先按服务器已有 CUDA 版本安装 PyTorch，再安装 DiffSynth 相关依赖。

基本要求：

- `torch`
- `transformers`
- `modelscope`
- `safetensors`
- `einops`
- `Pillow`
- 仓库内 `diffsynth` 可以被 Python import

可以先测试：

```bash
python - <<'PY'
import torch
import diffsynth
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("diffsynth import ok")
PY
```

## 4. 模型权重路径

默认脚本假设模型权重路径是：

```bash
/tmp/DiffSynth-Studio/models
```

如果模型放在其他路径，需要修改启动脚本中的参数：

```bash
--model-base-path /tmp/DiffSynth-Studio/models
```

改成真实路径，例如：

```bash
--model-base-path /data/models
```

脚本还会使用：

```bash
export MODELSCOPE_CACHE=/tmp/modelscope-cache
export HF_HOME=/tmp/hf-home
```

如服务器有统一缓存目录，可以改成对应路径。

## 5. 启动训练

### 5.1 推荐先跑单层训练

```bash
cd z-img-indexer-test
export CUDA_VISIBLE_DEVICES=0
export MODELSCOPE_CACHE=/tmp/modelscope-cache
export HF_HOME=/tmp/hf-home

bash experiments/z_image_indexer/run_train_csa_m2_topk64.sh
```

训练输出目录：

```text
experiments/z_image_indexer/results/train_csa_layer12_m2_topk64/
```

关键输出：

- `csa_indexer_distill.pt`
- `metrics.json`
- `summary.json`
- `run_config.json`

### 5.2 启动双层训练

```bash
bash experiments/z_image_indexer/run_train_csa_multilayer_l12_l13_m2_topk64.sh
```

训练输出目录：

```text
experiments/z_image_indexer/results/train_csa_layers12_13_m2_topk64/
```

关键输出：

- `csa_multilayer_indexer_distill.pt`
- `metrics.json`
- `summary.json`
- `run_config.json`

### 5.3 启动全层训练

```bash
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_bs4.sh
```

训练输出目录：

```text
experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_bs4/
```

关键输出：

- `csa_multilayer_indexer_distill.pt`
- `metrics.json`
- `summary.json`
- `run_config.json`

## 6. 全层训练的直接 Python 命令

如果不想使用 shell 脚本，可以直接执行：

```bash
python experiments/z_image_indexer/train_csa_multilayer_indexer.py \
  --model-base-path /tmp/DiffSynth-Studio/models \
  --prompt-file experiments/z_image_indexer/configs/default_prompts_train.txt \
  --output-dir experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_bs4 \
  --steps 1000 \
  --height 512 \
  --width 512 \
  --num-inference-steps 4 \
  --layer-ids all \
  --batch-size 4 \
  --compression-rate 2 \
  --compressed-topk 64 \
  --rank 128 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --recall-k 16 \
  --log-every 10 \
  --seed 42 \
  --device cuda
```

常用调整：

```bash
--steps 50
```

用于 smoke run。

```bash
--batch-size 2
```

用于显存不足时降 batch。

```bash
--layer-ids 0-14
```

用于只训练前半部分层。

```bash
--layer-ids 15-29
```

用于只训练后半部分层。

## 7. 训练后如何判断是否正常

训练结束后先看：

```bash
cat experiments/z_image_indexer/results/train_csa_all_layers_m2_topk64_bs4/summary.json
```

重点字段：

- `initial_loss`
- `final_loss`
- `initial_layer_recalls`
- `final_layer_recalls`
- `batch_size`
- `layer_ids`

正常情况：

- `summary.json` 能生成
- `metrics.json` 有每一步记录
- 大多数目标层的 `final_layer_recalls` 高于初始值

需要谨慎的情况：

- `final_loss` 高于 `initial_loss`
- 只有少数层 recall 提升
- 训练速度极慢或显存频繁 OOM

这些不一定表示脚本失败，但说明该配置还需要继续调参。

## 8. 训练后评测

当前已具备单层训练后的评测脚本：

```bash
bash experiments/z_image_indexer/run_eval_csa_m2_topk64.sh
```

输出目录：

```text
experiments/z_image_indexer/results/eval_csa_layer12_m2_topk64/
```

关键输出：

- `records.json`
- `summary.json`
- dense / sparse 生成图
- `prompt_*_compare_csa.png` 对比图

注意：

- 全层训练后的完整替换评测还需要继续补齐对应 eval 入口。
- 因此全层训练完成后，第一步先看训练指标；正式判断质量/速度收益需要后续全层推理替换评测。

## 9. 给其他人或 AI 的最短执行说明

可以直接把下面这段给训练服务器上的执行者：

```text
请在训练服务器上 clone 仓库 https://github.com/ZeroSky345/z-img-indexer-test，
进入仓库根目录后，确认 Python 环境可以 import torch 和 diffsynth，
确认 Z-Image-Turbo 模型权重路径为 /tmp/DiffSynth-Studio/models。

先运行：
bash experiments/z_image_indexer/run_train_csa_m2_topk64.sh

如果单层训练正常，再运行：
bash experiments/z_image_indexer/run_train_csa_multilayer_l12_l13_m2_topk64.sh

如果需要测试全部层训练，再运行：
bash experiments/z_image_indexer/run_train_csa_all_layers_m2_topk64_bs4.sh

如果全层训练 OOM，把脚本里的 --batch-size 4 改成 2；仍然 OOM 就改成 1。
训练结束后返回对应 results 目录下的 summary.json、metrics.json 和 checkpoint 路径。
```

## 10. 当前建议

正式迁移到新机器时，建议执行顺序是：

1. 单层训练 smoke run：`steps=50`
2. 单层训练完整 1000 步
3. 双层训练 1000 步
4. 全层训练 smoke run：`steps=50, batch_size=4`
5. 如果显存稳定，再跑全层 1000 步

不要一上来直接跑更高分辨率、多卡联合训练或主模型联合训练。当前代码主要验证的是 indexer-only 蒸馏路线。
