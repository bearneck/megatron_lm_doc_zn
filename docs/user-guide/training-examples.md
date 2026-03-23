<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 训练示例

通过这些实用示例开始使用 Megatron Core 进行训练。

## 简单训练示例

最简单的入门方式是使用模拟数据的基本训练循环：

```bash
# 在 2 个 GPU 上使用模拟数据进行分布式训练
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

此示例：
- 在 2 个 GPU 上运行
- 使用生成的模拟数据（无需数据准备）
- 演示基本的分布式训练设置
- 非常适合测试您的安装

## LLaMA-3 训练示例

### LLaMA-3 8B 使用 FP8

在 8 个 GPU 上使用 FP8 混合精度训练 LLaMA-3 8B 模型：

```bash
./examples/llama/train_llama3_8b_h100_fp8.sh
```

**配置：**
- 8 个 GPU
- FP8 混合精度（需要 Hopper/Ada/Blackwell GPU）
- 用于快速测试的模拟数据

### 自定义 LLaMA 训练

使用您自己的数据进行训练：

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --train-iters 100000 \
    --lr 3.0e-4 \
    --min-lr 3.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --data-path /path/to/your/preprocessed_data \
    --split 949,50,1 \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 1000
```

## GPT-3 训练示例

训练一个 GPT-3 风格的模型：

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --train-iters 100000 \
    --lr 1.5e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 1000 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --fp16 \
    --data-path /path/to/preprocessed_data \
    --split 949,50,1 \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints
```

## 关键训练参数

### 模型架构

| 参数 | 描述 |
|----------|-------------|
| `--num-layers` | Transformer 层数 |
| `--hidden-size` | 隐藏层维度大小 |
| `--num-attention-heads` | 注意力头数量 |
| `--seq-length` | 训练序列长度 |
### 训练配置

| 参数 | 描述 |
|----------|-------------|
| `--micro-batch-size` | 每个 GPU 的批大小 |
| `--global-batch-size` | 所有 GPU 上的总批大小 |
| `--train-iters` | 训练迭代次数 |

### 学习率

| 参数 | 描述 |
|----------|-------------|
| `--lr` | 峰值学习率 |
| `--min-lr` | 最小学习率 |
| `--lr-decay-style` | 学习率调度策略（余弦、线性、常数） |
| `--lr-warmup-iters` | 预热迭代次数 |

### 混合精度

| 参数 | 描述 |
|----------|-------------|
| `--fp16` | FP16 混合精度 |
| `--bf16` | BF16 混合精度（推荐） |
| `--fp8-hybrid` | FP8 混合精度（Hopper/Ada/Blackwell 架构） |

### 数据与检查点

| 参数 | 描述 |
|----------|-------------|
| `--data-path` | 预处理数据的路径 |
| `--split` | 训练/验证/测试集划分（例如：949,50,1） |
| `--save` | 检查点保存目录 |
| `--load` | 检查点加载目录 |
| `--save-interval` | 每 N 次迭代保存一次检查点 |

## 后续步骤

- **优化性能**：查看[高级功能](features/index.md)了解 FSDP、分布式优化器及其他优化方法
- **扩大规模**：学习[并行策略](parallelism-guide.md)，以便在更多 GPU 上训练更大的模型
- **准备数据**：按照[数据准备](data-preparation.md)指南处理您自己的数据集