<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 并行策略指南

Megatron Core 支持多种并行策略，这些策略可以组合使用，以在数千个 GPU 上高效训练从数十亿到数万亿参数的模型。

## 概述

| 策略 | 并行化的对象 | 最佳适用场景 |
|----------|---------------------|----------|
| **数据并行 (DP)** | 批次维度 | 标准训练，最常见 |
| **张量并行 (TP)** | 单个层 | 大型层，GPU 内存限制 |
| **流水线并行 (PP)** | 模型深度 | 非常深的模型 |
| **上下文并行 (CP)** | 序列长度 | 长序列（8K+ tokens） |
| **专家并行 (EP)** | MoE 专家 | 专家混合模型 |

## 数据并行 (DP)

在 GPU 间复制模型，并分割批次。

### 标准数据并行 (DDP)

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --data-parallel-sharding-strategy no_shard
```

每个 GPU 拥有模型的完整副本，并处理批次的一部分。

### 全分片数据并行 (FSDP)

分片模型参数、梯度和优化器状态以减少内存占用：

```bash
# Megatron FSDP（比 PyTorch FSDP2 快约 15%）
--use-megatron-fsdp \
--data-parallel-sharding-strategy optim_grads_params
```

**分片策略：**
- `optim` - 仅分片优化器状态 (ZeRO-1)
- `optim_grads` - 分片梯度 + 优化器 (ZeRO-2)
- `optim_grads_params` - 分片参数 + 梯度 + 优化器 (ZeRO-3)

## 张量并行 (TP)

在 GPU 间分割单个模型层。推荐用于大型隐藏维度。

```bash
--tensor-model-parallel-size 4  # 4 路张量并行
--sequence-parallel              # 启用序列并行（推荐）
```

**何时使用：**
- 模型层无法放入单个 GPU
- 大型隐藏维度（4096+）
- 通常与 DP 和 PP 结合使用

## 流水线并行 (PP)

在 GPU 间垂直（按深度）分割模型层。

```bash
--pipeline-model-parallel-size 8              # 8 个流水线阶段
--num-layers-per-virtual-pipeline-stage 4     # 用于负载均衡的虚拟流水线
```

**何时使用：**
- 非常深的模型（50+ 层）
- 与 TP 结合用于大型模型
- 有助于在 GPU 间分配内存

## 上下文并行 (CP)

在 GPU 间分割长序列，以实现高效的长上下文训练。

```bash
--context-parallel-size 2           # 2 路上下文并行
--cp-comm-type p2p                  # 通信类型
```

**何时使用：**
- 长序列（8K+ tokens）
- 减少激活内存
- 可与 TP、PP、DP 结合使用
**→ [上下文并行深度解析](features/context_parallel.md)** - 包含性能分析的详细指南

## 专家并行 (EP)

在混合专家模型中，将专家分布到多个 GPU 上。

```bash
--expert-model-parallel-size 8  # 8路专家并行
--num-experts 64                # 每个MoE层有64个专家
--moe-grouped-gemm              # 优化专家计算
```

**重要提示：** 当将 EP 与 TP 结合使用时，**必须启用序列并行**：

```bash
--tensor-model-parallel-size 4
--expert-model-parallel-size 8
--sequence-parallel  # 使用 TP + EP 时必须启用
```

## 并行策略选择指南

基于 [NVIDIA NeMo 生产环境配置](https://github.com/NVIDIA/NeMo/tree/main/scripts/performance/recommended_model_configs) 的推荐配置：

### 语言模型

| 模型 | 规模 | GPU数量 | TP | PP | CP | EP | 配置说明 |
|-------|------|------|----|----|----|----|---------------------|
| **LLaMA-3** | 8B | 8 | 1 | 1 | 2 | 1 | CP=2 用于长上下文 (8K 序列长度) |
| **LLaMA-3** | 70B | 64 | 4 | 4 | 2 | 1 | 针对70B规模的平衡TP+PP配置 |
| **LLaMA-3.1** | 405B | 1024 | 8 | 8 | 2 | 1 | 3D并行 (TP+PP+CP) |
| **GPT-3** | 175B | 128-512 | 4 | 8 | 1 | 1 | 标准大模型配置 |

### 混合专家模型

| 模型 | 规模 | GPU数量 | TP | PP | CP | EP | 配置说明 |
|-------|------|------|----|----|----|----|---------------------|
| **Mixtral** | 8x7B | 64 | 1 | 4 | 1 | 8 | EP=8 对应8个专家 |
| **Mixtral** | 8x22B | 256 | 4 | 4 | 1 | 8 | 大型MoE模型的TP+PP+EP配置 |
| **DeepSeek-V3** | 671B | 1024 | 2 | 16 | 1 | 64 | 拥有256个专家的大规模MoE模型 |

## 组合策略

### GPU总数计算

GPU总数计算公式如下：

```
GPU总数 = TP × PP × CP × EP × DP
```

### 示例：在64个GPU上运行LLaMA-3 70B

```bash
# TP=4, PP=4, CP=2, DP=2 => 4 × 4 × 2 × 2 = 64 GPUs
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 4 \
    --context-parallel-size 2 \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 8192 \
    --micro-batch-size 1 \
    --global-batch-size 512 \
    --bf16
```

## 性能优化

### 通信重叠

启用通信与计算的重叠：

```bash
--overlap-grad-reduce      # 梯度归约与反向传播重叠
--overlap-param-gather     # 参数收集与前向传播重叠
--tp-comm-overlap          # 重叠TP通信
```

### 分布式优化器

推荐用于所有多GPU训练：

```bash
--use-distributed-optimizer
```

优势：
- 更快的检查点保存
- 与FSDP结合时减少内存占用
- 大规模训练时性能更佳

### 序列并行

使用TP时始终启用：

```bash
--sequence-parallel
```

通过在LayerNorm和Dropout中分片序列维度，减少激活内存。

## 选择正确的策略

### 从简单开始
1. 首先仅使用**数据并行**
2. 如果模型放不下，添加**张量并行**
3. 对于非常大的模型，添加**流水线并行**
4. 对于长序列，添加**上下文并行**
### 内存限制
- 使用 **FSDP** 来减少每个 GPU 的内存占用
- 使用 **TP** 来拆分大型层
- 使用 **PP** 来拆分模型深度
- 在极端情况下启用 **激活检查点**

### 通信瓶颈
- 降低 **TP** 并行度（会增加每个 GPU 的内存占用）
- 增加 **PP** 并行度（可能会降低效率）
- 对于长序列，使用 **CP** 而不是更大的 TP

## 后续步骤

- **API 参考**：请参阅 [张量并行](../api-guide/core/tensor_parallel.md) 和 [流水线并行](../api-guide/core/pipeline_parallel.md) API 文档
- **高级功能**：探索 [Megatron FSDP](features/custom_fsdp.md) 和 [分布式优化器](features/dist_optimizer.md)
- **性能调优**：查看 [NVIDIA NeMo 性能指南](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)