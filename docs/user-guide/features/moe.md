<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 混合专家模型（Megatron Core MoE）

Megatron Core MoE 是一个用于训练大规模混合专家模型的生产就绪框架，它提供了基础架构、性能优化和行业最佳实践，指导着整个行业的 MoE 框架开发。

## 最新动态
关于最新功能和架构，请参阅 [MCore 开发路线图](https://github.com/NVIDIA/Megatron-LM/issues/1729)。

### 🔥 [MCore 开发版] (2026/01)
- 🚀 支持流水线感知的细粒度激活卸载
- 🚀 支持 Qwen3-Next 模型
- 🚀 支持 DeepSeek-V3.2 模型
- 🚀 支持 Muon 和分层分布式优化器
- 🚀 支持具有细粒度作用域的 CUDA Graph

### 🔥 [MCore v0.15] (2025/11)
- 🚀 为 Flex Dispatcher 添加 HybridEP 后端（支持 GB200, B200, H100）
- 🚀 支持 MoE 模型使用 EP 的 FSDP

### 🔥 [MCore v0.14] (2025/09)
- 🚀 批处理级重叠以隐藏 EP-A2A 通信 (--overlap-moe-expert-parallel-comm --delay-wgrad-compute)
- 🚀 支持细粒度重计算的 FP8
- 🚀 MoE 模型的路由器融合内核 (--moe-router-fusion)
- 🚀 为 MTP 和 MLA 支持上下文并行

### 🔥 [MCore v0.13] (2025/07)
- 🚀 支持优化器状态使用 bf16 数据类型，以在 TransformerEngine 中使用精度感知优化器 (--use-precision-aware-optimizer)
- 🚀 支持具有自定义流水线布局的灵活非对称虚拟流水线并行 (--pipeline-model-parallel-layout)
- 🚀 为 MoE 模型添加混合分片数据并行支持 (--num-distributed-optimizer-instances)
- 🚀 支持细粒度重计算以减少激活内存 (--recompute-modules 配合 --recompute-granularity selective)
- 🚀 通过将概率乘法从反置换移动到 GroupedMLP 的激活函数，实现内存高效的令牌置换。

### 🔥 [MCore v0.12] (2025/05)
- 🚀 支持 DeepSeek 的 DeepEP 以实现高效的令牌分发 (--moe-token-dispatcher-type flex --moe-enable-deepep)
- 🚀 支持多令牌预测 (--mtp-num-layers 1)
- 🚀 为无丢弃 MoE 模型支持仅捕获注意力的 CUDA Graph (--te-rng-track --external-cuda-graph --cuda-graph-scope attn)

## MCore MoE 支持的功能和架构概述

### 模型支持
- ✅ **DeepSeek**
  - ✅ DeepSeek-V2
  - ✅ DeepSeek-V3，包括 MTP
- ✅ **Qwen**
  - ✅ Qwen2-57B-A14B
  - ✅ Qwen3-30B-A3B
  - ✅ Qwen3-235B-A22B
- ✅ **Mixtral**
  - ✅ Mixtral-8x7B
  - ✅ Mixtral-8x22B

### 核心 MoE 功能
- ✅ 无令牌丢弃 MoE - 无需令牌丢弃的高级路由
- ✅ 支持灵活 K 值选择的 Top-K 路由器
- ✅ 用于优化专家利用率的负载均衡损失

### 高级并行
- ✅ 与 3D 并行集成的专家并行
- ✅ 完整的并行组合：支持 EP + DP + TP + PP + SP
- ✅ 用于长序列 MoE 训练的上下文并行
- ✅ 用于高效大规模 MoE 模型训练的并行折叠异构并行映射
- ✅ 用于 MoE 的分布式优化器

### 性能优化
- ✅ 内存高效的令牌置换
- ✅ 细粒度重计算
- ✅ 为混合线性注意力支持 MLA TP
- ✅ GroupedGEMM 和 GA 融合
- ✅ DP/PP/TP 通信重叠
- ✅ 重叠的共享专家执行
- ✅ 路由器融合优化
- ✅ 令牌（反）置换融合内核
- ✅ cuDNN 融合注意力集成
### 硬件与精度支持
- ✅ 支持 H100 和 B200 的 DeepEP
- ✅ 支持 FP8/MXFP8 的 GroupedGEMM
- ✅ 支持 FP8 权重与 BF16 优化器状态
- ✅ 全面支持 FP8 训练

### 开发者体验
- ✅ 包含预训练最佳实践的 MoE 模型库
- ✅ 支持 MoE 模型的分布式检查点
- ✅ 支持模型扩展的升级利用
- ✅ 用于生态系统兼容的 MCore2HF 转换器
- ✅ 用于详细监控的逐层日志记录
- ✅ 运行时升级利用能力

## 快速入门指南

### Megatron-LM 中的基础 MoE 训练

要训练一个包含 8 个专家和辅助损失函数的 top-2 MoE 模型，请将以下参数添加到您的 Megatron 训练脚本中：

```bash
## Set MoE Hidden site
--num-experts 8
--moe-shared-expert-intermediate-size: 2048
## Set router config
--moe-router-load-balancing-type aux_loss
--moe-router-topk 2
--moe-aux-loss-coeff 1e-2
## Set token dispatcher
--moe-token-dispatcher-type alltoall
```

每个功能的详细文档可在 [功能文档](#feature-documentation) 部分找到。

### 使用预定义配置训练流行的 MoE 模型
我们在 [Megatron-MoE-Model-Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo/tree/main) 仓库中提供了一些预定义配置，用于训练流行的 MoE 模型。您可以将它们作为配置训练脚本的参考。目前我们已添加了 Mixtral 8x7B、Mixtral 8x22B、DeepSeek-V3、Qwen3-30B-A3B、Qwen3-235B-A22B 的配置。

### 通用性能优化技巧
#### 训练参数
以下标志是通用性能标志，几乎可以帮助所有工作负载实现更高的性能。请检查您的训练脚本是否已启用所有这些标志。

```bash
## Enable DeepEP token dispatcher
--moe-token-dispatcher-type flex
--moe-flex-dispatcher-backend deepep
## Enable GroupedGEMM
--moe-grouped-gemm
## Enable fusion kernels
--moe-router-fusion
--moe-permute-fusion
--cross-entropy-loss-fusion
--cross-entropy-fusion-impl te

## Communication optimization
--use-distributed-optimizer
--overlap-param-gather
--overlap-grad-reduce
--tp-comm-overlap

## Enable manual gc to prevent python jitter
--manual-gc: true
--manual-gc-interval: 10
```
#### 环境变量

以下是一些可能有用的环境变量。
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Enable expandable segments to prevent memory fragmentation
export NCCL_NVLS_ENABLE=0 # Disable NVLS to prevent memory overhead
```
#### 依赖项
- 使用最新版本的 [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)。
- 使用最新的 [NGC PyTorch Docker 镜像](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

## 实现 MoE 训练高性能的最佳实践

分布式训练涉及**通信**、**内存**和**计算**之间复杂的权衡，因此找到最优的并行配置具有挑战性。本节提供了一个系统化的工作流程，帮助您为您的模型和硬件找到最佳的并行映射方案。
### 步骤 1：在 GPU 显存容量下找到可行的并行映射
为了找到最佳的并行映射，我们首先需要知道在 GPU 显存容量下模型可行的并行映射。
内存消耗由三部分组成：
- 激活内存
- 权重和梯度内存
- 优化器状态内存
不同的并行策略会以不同的方式对这些张量内存进行分片。

| 并行策略 | 峰值激活内存          | 权重内存  | 优化器状态                  | 通信（每层） |
|:-----------------:|:-------------------------------:|:--------------:|:---------------------------------:|:-------------------------:|
| TP                | 1/N (启用 SP 时)                | 1/N            | 1/N                               |        高               |
| EP                | ~1 (随 EP 平衡而变化)   | MoELayer 中为 1/N| 1/N                               |       中等              |
| PP                | 1 (使用虚拟流水线时 >1)    | 1/N            | 1/N                               |       中等              |
| CP                | 1/N                             | 1              | 1/N (使用分布式优化器时)  |       中等              |
| DP                | 1                               | 1              | 1/N (使用分布式优化器时)  |        低                |

我们提供了 `--fake-init-process-group` 参数，用于在单个 GPU 上模拟分布式训练。这对于在 GPU 显存容量下找到可行的并行映射非常有用。详细用法请参见 https://github.com/NVIDIA/Megatron-LM/pull/2254。

### 步骤 2：选择最优并行策略

最优的并行配置因**模型架构**、**序列长度**和**硬件平台**而异。以下是一些通用指南，可帮助您实现高吞吐量。

#### 指南 1：最小化模型并行，最大化数据并行

| 方面 | 建议 |
|--------|----------------|
| **目标** | 在避免 OOM 的同时，尽可能保持 TP/EP/PP 较小 |
| **原因** | 模型并行会引入通信开销，损害性能 |
| **方法** | 使用分布式优化器 (`--use-distributed-optimizer`) 将优化器状态分片到 DP 进程上，释放内存以支持更大的 DP 规模 |

#### 指南 2：将 EP 和 TP 通信保持在 NVLink 域内

| 方面 | 建议 |
|--------|----------------|
| **目标** | 确保 EP×TP 适合单个节点（通常为 8 个 GPU） |
| **原因** | EP 和 TP 是通信密集型的；NVLink 提供的带宽远高于跨节点互连 |
| **扩展** | 当扩展到单个节点之外时，优先使用 PP 而不是跨节点扩展 TP/EP |

**注意：**
对于像 DeepSeek-V3 这样非常大的 MoE 模型，EP 通信可能超过 NVLink 带宽。在这种情况下，请考虑使用 1F1B A2A 重叠来重叠 EP 通信。

#### 指南 3：使用流水线并行进行多节点扩展
| 方面 | 建议 |
|--------|----------------|
| **目标** | 使用 PP 跨节点分布层，同时保持 EP×TP 在 NVLink 范围内 |
| **VPP** | 当 `PP ≥ 2` 时，启用虚拟流水线并行以减少流水线气泡 |
| **配置** | 设置 `--num-layers-per-virtual-pipeline-stage` 以控制 VPP 大小 |

**VPP 大小调优：**
- 有效值：`num_layers / PP_size` 的所有除数
- 示例：`num_layers=24, PP=4` → 有效 VPP 大小：`{1, 2, 3, 6}`
- 权衡：更大的 VPP = 更少的气泡但更多的 P2P 通信
- 建议：中间值通常能提供最佳平衡

#### 指南 4：对于专家层，优先选择 EP 而非 TP

| EP 优势 | 详情 |
|---------------|---------|
| **更好的 GEMM 效率** | 更大的本地矩阵尺寸提高了 GPU 利用率 |
| **更低的通信开销** | 对于 MoE 层，EP 比 TP 的通信开销更小 |
| **更简单的计算图** | 更容易实现通信与计算的重叠 |
| **令牌置换** | 当 `EP = num_experts` 时，本地令牌置换被消除 |

**示例：** 对于 Mixtral 8x7B，`EP8×TP1` 优于 `EP4×TP2`。

#### 指南 5：为长序列启用上下文并行

| 方面 | 建议 |
|--------|----------------|
| **何时使用** | 序列长度 ≥ 8K 令牌 |
| **关键因素** | CP 效率取决于通信与计算的重叠程度 |
| **配置** | 设置 `--context-parallel-size` 以跨 GPU 分区序列 |

### 步骤 3：基于性能分析瓶颈启用性能优化功能

建立可工作的并行配置后，分析您的训练以识别瓶颈并应用有针对性的优化。

#### 内存瓶颈

**症状**：被迫使用完全重计算或过大的并行度以避免 OOM。

**解决方案**：
| 优化 | 开销 | 配置 | 参考 |
|--------------|----------|--------|---------|
| 选择性重计算 | 低 | `--recompute-granularity selective --recompute-modules ...` | [细粒度重计算](#fine-grained-recomputation) |
| 激活卸载 | 中 | `--fine-grained-activation-offloading --offload-modules ...` | [细粒度激活卸载](#fine-grained-activation-offloading) |
| 优化器卸载 | 中 | `--optimizer-cpu-offload` | --- |

#### 通信瓶颈

**症状**：性能分析显示在集体操作上花费了大量时间。

**解决方案**：识别哪种通信是瓶颈并启用相应的重叠：
| 通信类型 | 重叠配置 |
|--------------------|----------------|
| DP 梯度规约 | `--overlap-grad-reduce` |
| DP 参数收集 | `--overlap-param-gather` |
| TP 通信 | `--tp-comm-overlap` |
| EP All-to-All | `--overlap-moe-expert-parallel-comm --delay-wgrad-compute` |
| PP 发送/接收 | 使用 `--num-layers-per-virtual-pipeline-stage` 启用 VPP |

#### CPU 开销瓶颈

**症状**：Nsight Systems 时间线显示 GPU 内核之间存在间隙，CPU 无法足够快地启动内核。
**解决方案**：
| 优化项 | 配置 |
|--------------|--------|
| 禁用 Python GC | `--manual-gc --manual-gc-interval 100` |
| 启用 CUDA Graphs | `--cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess` |
| 减少内核启动 | 降低 TP 大小或增加微批次大小 |

#### 计算瓶颈

**症状**：尽管没有通信或 CPU 瓶颈，但 GPU 利用率很低。

**解决方案**：
| 优化项 | 配置 |
|--------------|--------|
| 启用内核融合 | `--moe-router-fusion --moe-grouped-gemm --moe-permute-fusion` |
| 使用 FP8 精度 | `--fp8-format e4m3 --fp8-recipe blockwise` |


## 功能文档

### 路由器和负载均衡

路由器决定哪些专家处理每个令牌。一个轻量级的 MLP 为每个令牌打分，并应用 `softmax` 或 `sigmoid` 来计算路由概率。然后，路由器为每个令牌选择 Top-K 个专家。

> **注意**：路由器 logits 最好保持为 **FP32** 或 **FP64**，而不是 BF16，可通过 `--moe-router-dtype fp32` 设置。在专家数量较多时，FP32 精度能提供更好的准确性，因为专家的输出隐藏状态会乘以路由分数并累加得到最终输出。

#### 路由器类型

| 路由器类型 | 描述 | 配置 |
|-------------|-------------|----------|
| **Top-K 路由器** | 标准路由，可配置 K 值，使用 softmax 计算概率 | --moe-router-topk 8 |
| **分组 Top-K 路由器** | 选择 Top-K 个专家组，然后在选定的组内路由专家 | --moe-router-num-groups 8 --moe-router-group-topk 4 |
| **路由器分数函数** | 用于根据路由器输出 logits 计算概率的分数函数 | --moe-router-score-function softmax/sigmoid |

#### 负载均衡策略

| 策略 | 描述 | 配置 |
|----------|-------------|--------|
| **aux_loss** | 用于在微批次上平衡专家使用的辅助损失 | `--moe-router-load-balancing-type aux_loss` |
| **seq_aux_loss** | 用于在每个序列上平衡专家使用的序列级辅助损失 | `--moe-router-load-balancing-type seq_aux_loss` |
| **global_aux_loss** | 用于在所有 rank 的全局批次上平衡专家使用的全局辅助损失 | `--moe-router-load-balancing-type global_aux_loss` |
| **sinkhorn** | 用于平衡专家使用的最优传输公式 | `--moe-router-load-balancing-type sinkhorn` |
| **aux loss free** | 基于动态偏置的负载均衡策略，无需辅助损失 | `--moe-router-enable-expert-bias --moe-router-bias-update-rate 1e-3`|
| **none** | 无负载均衡 | `--moe-router-load-balancing-type none` |

### 令牌分发

路由之后，令牌被**分发**到托管指定专家的 GPU 上。专家计算完成后，令牌被发送回来并**组合**以恢复原始序列。

| 分发器 | 描述 | 最佳适用场景 | 配置 |
|------------|-------------|----------|--------|
| **alltoall** | 基于 NCCL 的 All-to-All 通信，用于令牌交换 | 标准 EP > 1 的设置 | `--moe-token-dispatcher-type alltoall` |
| **FlexDispatcher with [DeepEP](https://github.com/deepseek-ai/DeepEP) backend** | 在跨节点通信时移除冗余令牌，将节点内/节点间通信融合到单个内核中 | 跨节点 EP，细粒度 MoE (DeepSeek-V3) | `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep` |
| **FlexDispatcher with [HybridEP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep) backend** | NVIDIA 优化的分发器，使用 TMA 和 IBGDA，占用更少的 SM，原生支持 MNNVL | GB200 NVL72，多节点 NVLink | `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep` |
| **allgather** | 将所有令牌收集到每个 GPU，无 GPU 间令牌移动 | 仅 TP 的设置，小 EP，大 Top-K | `--moe-token-dispatcher-type allgather` |
### 升级回收
使用 `--moe-use-upcycling` 来启用升级回收功能。该功能会从 `--load` 目录加载稠密模型，在运行时将其转换为 MoE 模型，然后开始训练。转换后的模型在训练开始前会保存到 `--save` 路径。升级回收功能基于分布式检查点技术构建，支持与现有稠密检查点不同的并行模式，例如在升级回收期间使用任意的专家并行度。

除了默认的升级回收策略，我们还支持**细粒度升级回收策略**，这是来自[我们近期研究工作](https://arxiv.org/abs/2410.07524)的一种更先进的升级回收策略。对于默认策略，我们将现有的 MLP 复制到多个专家，每个专家都从该 MLP 的一个副本开始。对于细粒度策略，我们使用 `--moe-upcycling-granularity` 来指定专家隐藏层大小比原始稠密 FFN 隐藏层大小小多少倍。要使用细粒度升级回收策略，请将 `--moe-upcycling-granularity` 设置为一个正整数。如果此参数设置为 1，则表示使用默认的升级回收策略。

注意：MoE 模型结构是通过脚本参数定义的。所有与 MoE 相关的参数（例如 `--num-experts`）都可以自定义；但是，其他模型结构参数必须与稠密模型的参数保持一致。对于细粒度升级回收策略，MoE 的 FFN 隐藏层大小应设置为稠密 FFN 隐藏层大小除以 `--moe-upcycling-granularity`。

## 训练优化
MoE 训练面临三个基本的性能瓶颈：**内存墙**、**通信墙**和**计算效率墙**。以下优化措施旨在应对这些挑战。

### MoE 并行折叠
**传统方法的问题：**
- 先前的 MoE 框架限制 **EP ≤ DP**（专家并行度必须是数据并行度的子组），这严重限制了可扩展性。
- 对注意力层和 MoE 层应用相同的 TP/CP 是次优的：
  - 高 TP 对注意力层有益，但对 MoE 层有害（每个专家的维度较小使得 TP 开销过大）
  - 高 CP 对长上下文注意力层有益，但对 MoE 层不必要（token 是独立处理的）

**MoE 并行折叠**是 Megatron Core 的解决方案，它**解耦了注意力层和 MoE 层的并行策略**：

| 并行组 | 注意力层 | MoE 层 |
|-------------------|------------------|------------|
| **维度** | TP × CP × DP × PP | ETP × EP × EDP × PP |

#### 主要优势

1. **打破 EP ≤ DP 的限制**
   - 传统方式：TP=4, CP=2, DP=8, PP=4 → 最大 EP=8
   - 使用折叠：注意力层配置相同，但 MoE 层使用 ETP=1, EP=64, EDP=1 → 专家并行度提升 8 倍

2. **降低最低 GPU 需求**
   - 传统方式 CP=8, EP=8 至少需要 64 个 GPU
   - 使用折叠：CP 和 EP 被折叠在一起，仅需 8 个 GPU

3. **实现独立优化**
   - 对注意力层使用高 TP（内存效率高）
   - 对 MoE 层使用 ETP=1（GEMM 效率更高，通信更少）
4. **保持 NVLink 域内的高带宽通信**
   - CP 和 EP 通信均可保持在 NVLink 域内

> **参考**：[MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training](https://arxiv.org/abs/2504.14960)

### 内存优化

内存优化对于大规模 MoE 训练至关重要，因为 MoE 模型会保留所有专家参数，尽管每个 token 只激活其中的一个子集。

| 优化方法 | 描述 | 配置 |
|--------------|-------------|--------|
| **细粒度重计算** | 选择性地重新计算特定模块（例如 `mla_up_proj`、`layernorm`、`moe_act`），而不是整个层 | `--recompute-granularity selective --recompute-modules mla_up_proj layernorm moe_act` |
| **细粒度激活卸载** | 将激活值卸载到 CPU 内存，使 D2H/H2D 传输与计算重叠 | 参见 `docs/source/api-guide/fine_grained_activation_offloading.md` |
| **精度感知优化器** | 以 BF16 而非 FP32 存储优化器状态（exp_avg, exp_avg_sq），将优化器内存减少 50% | `--use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16` |
| **优化器卸载** | 将优化器状态卸载到 CPU 内存。 | `--optimizer-cpu-offload` |

#### 细粒度重计算
还支持一种新的输出丢弃检查点方法。该方法在前向传播期间丢弃某些子模块的输出内存，并在反向传播期间重新计算它们，与标准检查点方法相比可以节省内存。可以使用 `--recompute-granularity selective --recompute-modules [submodule1, submodule2, ...]` 参数为特定子模块启用此功能。支持的子模块有：

* `moe_act`：重新计算 GroupedMLP 激活函数。
* `layernorm`：重新计算 input_layernorm 和 pre_mlp_layernorm（当它们不是 `IdentityOp` 时）。
* `mla_up_proj`：重新计算 MLA 上投影和 RoPE 应用部分。
* `core_attn`：重新计算核心注意力子模块（使用标准检查点而非输出丢弃）。
* `mlp`：重新计算密集 MLP 子模块（使用标准检查点而非输出丢弃），这对于像 DeepSeek-V3 这样的混合模型很有用。
* `moe`：重新计算 MoE 层子模块（使用标准检查点而非输出丢弃）。

#### 细粒度激活卸载

与重计算（用计算换取内存）不同，卸载是**用 GPU-CPU 带宽换取内存**：激活值在前向传播期间传输到 CPU，并在反向传播期间取回。关键是通过异步 D2H/H2D 传输将传输延迟隐藏在计算之后。

**主要特性：**
- **模块级粒度**：针对特定模块而非整个层
- **计算-卸载重叠**：通过独立的 CUDA 流进行异步传输
- **兼容 PP/VPP**：可与流水线并行和细粒度重计算协同工作

**用法**
```bash
--fine-grained-activation-offloading
--offload-modules expert_fc1 moe_act # Choices: attn_norm, core_attn, attn_proj, mlp_norm, expert_fc1, moe_act
```
更多详细信息，请参阅 `docs/source/api-guide/fine_grained_activation_offloading.md`

### 通信优化

分布式训练引入了来自各种并行策略的通信开销。Megatron Core 支持通信与计算重叠，以隐藏延迟并提高吞吐量。

#### 数据并行 (DP) 通信重叠

使用分布式优化器时，DP 引入了按 Transformer 层粒度分块的 **reduce-scatter**（梯度）和 **all-gather**（参数）通信。

| 优化项 | 描述 | 配置 |
|--------------|-------------|--------|
| **梯度规约重叠** | 将梯度 reduce-scatter 与反向计算重叠 | `--overlap-grad-reduce` |
| **参数收集重叠** | 将参数 all-gather 与前向计算重叠 | `--overlap-param-gather` |
| **BF16 梯度规约** | 以 BF16 而非 FP32 规约梯度以获得更好性能 | `--grad-reduce-in-fp32 false`（通过混合精度配置） |
| **FP8 参数收集** | 以 FP8 进行参数 all-gather，减少 50% 开销 | `--fp8-param-gather` |

#### 张量并行 (TP) 通信重叠

结合序列并行时，TP 引入了激活 all-gather 和 reduce-scatter 操作。通信以 **批量**（无依赖）或 **流水线**（有依赖）方式重叠。

| 优化项 | 描述 | 配置 |
|--------------|-------------|--------|
| **TP 通信重叠** | 启用批量和流水线式 TP 通信重叠 | `--tp-comm-overlap` |

> **要求**：`tensor_model_parallel_size >= 2` 且 `--sequence-parallel`

#### 流水线并行 (PP) 通信重叠

PP 在流水线阶段之间引入了 P2P 激活发送/接收。当启用 VPP 时，在 1F1B 流水线阶段中重叠是自动的。

| 优化项 | 描述 | 配置 |
|--------------|-------------|--------|
| **P2P 通信重叠** | 将 PP P2P 通信与非依赖计算重叠 | `--overlap-p2p-comm`（启用 VPP 时自动启用） |
| **VPP 以实现更好重叠** | 通过减少每个虚拟阶段的层数来增加重叠机会 | `--num-layers-per-virtual-pipeline-stage` |

#### 专家并行 (EP) 通信重叠

未经优化时，EP All-to-All 可能消耗 30-40% 的训练时间。这些功能可以隐藏或减少 EP 通信开销。

| 优化项 | 描述 | 配置 |
|--------------|-------------|--------|
| **EP A2A 重叠** | 通过合并相邻微批次的 FWD-BWD 过程，将 All-to-All 与计算重叠 | `--overlap-moe-expert-parallel-comm --delay-wgrad-compute` |
| **共享专家重叠** | 在 EP 令牌传输的同时并发运行共享专家计算 | `--moe-shared-expert-overlap` |

> **EP A2A 重叠的要求**：`expert_model_parallel_size > 1`, CUDA_DEVICE_MAX_CONNECTIONS > 1.

### 计算优化

细粒度 MoE 会产生许多小型操作，可能导致 GPU 资源利用不足。这些优化减少了内核启动开销并提高了 GPU 利用率。
| 优化 | 描述 | 配置 |
|--------------|-------------|--------|
| **分组 GEMM** | 将多个专家 GEMM 操作批量处理到单个内核调用中，提高 GPU 利用率 | `--moe-grouped-gemm` |
| **路由器融合** | 将路由器投影、top-k 选择、softmax 和辅助损失融合到更少的内核中 | `--moe-router-fusion` |
| **置换融合** | 将令牌置换/反置换操作融合到优化的单个内核中 | `--moe-permute-fusion` |
| **FP8 训练** | 在 Hopper/Blackwell GPU 上使用 FP8 Tensor Core 操作以获得更快的 GEMM | `--fp8 --fp8-recipe blockwise` |


### FP8 训练

FP8 训练在以下三个性能瓶颈方面均能带来益处：

| 瓶颈 | FP8 益处 | 影响 |
|------|-------------|--------|
| **内存** | 激活值减少 50% | 将线性层输入以 FP8 而非 BF16 存储 |
| **内存** | 消除 BF16 权重副本 | 原生 FP8 直接从 FP32 转换到 FP8 |
| **通信** | EP 分发量减少 50% | 以 FP8 而非 BF16 分发令牌 |
| **通信** | 参数 all-gather 减少 50% | 使用 FP8 主权重时（MXFP8 除外） |
| **计算** | 更快的 Tensor Core GEMM | 在 Hopper/Blackwell 上，FP8 操作比 BF16 更快 |

#### FP8 配方

| 配方 | 缩放粒度 | 格式 | 平台 | 使用场景 |
|--------|---------------------|--------|----------|----------|
| **Per-tensor** | 整个张量 | E4M3/E5M2 混合 | Hopper, Blackwell | 保守，初始实验 |
| **Blockwise** | 1×128 (激活值), 128×128 (权重) | E4M3 | Hopper | **生产验证** (DeepSeek-V3, Minimax-M2) |
| **MXFP8** | 1×32 | E4M3 + E8M0 缩放 | Blackwell | GB200 上的原生硬件支持 |

> **推荐**：在 Hopper 上使用 **blockwise FP8** 进行生产训练。它已在 DeepSeek-V3 级别的模型上进行了大规模验证。

#### MoE 专用 FP8 优化

| 优化 | 描述 | 配置 |
|--------------|-------------|--------|
| **路由图填充** | 填充路由图（而非令牌）以使 M 维度对齐到 16/32，避免 per-tensor 填充开销 | `--moe-router-padding-for-fp8` |
| **FP8 主权重** | 将 FP32 主权重直接转换为 FP8，消除 BF16 中间副本 | `--fp8-param-gather` (对于 MXFP8，需要额外添加 `--reuse-grad-buf-for-mxfp8-param-ag`) |


#### 配置示例

```bash
# Blockwise FP8 on Hopper (recommended for production)
--fp8-format e4m3
--fp8-recipe blockwise
--fp8-param-gather
--moe-router-padding-for-fp8

# MXFP8 on Blackwell
--fp8-format e4m3
--fp8-recipe mxfp8
--moe-router-padding-for-fp8
--fp8-param-gather
--reuse-grad-buf-for-mxfp8-param-ag
```

> **注意**：对于当前缩放的 blockwise 和 MXFP8 配方，训练损失曲线与 BF16 基线相比差异可忽略不计。


### CUDA Graph
CUDA Graph 功能可以通过 `--cuda-graph-impl` 选项启用。有两种实现方式：

1. `--cuda-graph-impl=local`: 使用 MCore 内部的 cuda graph 管理器捕获 cuda graph。
2. `--cuda-graph-impl=transformer_engine`: 使用 TE 的 `make_graphed_callables()` 接口捕获 cuda graph。
要使用 `--cuda-graph-impl=transformer_engine`，用户应在训练脚本中调用相关方法 `TECudaGraphHelper.create_cudagraphs()` 和 `TECudaGraphHelper.cuda_graph_set_manual_hooks()`。请参考 `megatron/training/training.py` 中的用法。

对于 MoE 模型，某些配置可能会阻止 CUDA Graph 捕获 MoE 层。具体来说，当未设置 `--moe-expert-capacity-factor` 和 `--moe-pad-expert-input-to-capacity` 时，产生的动态形状会使 MoE 层无法被捕获。在这种情况下，您仍然可以通过设置 `--cuda-graph-scope=attn` 来利用 CUDA Graphs 处理注意力层（`TransformerLayer._forward_attention()` 中的操作），同时保持 MoE 层（`TransformerLayer._forward_mlp()` 中的操作）不变。有关 `--cuda-graph-scope` 的更多用法，请参阅参数描述。

## MoE 参数参考
### 核心参数
| 参数 | 描述 | 默认值 |
|----------|-------------|---------|
| --num-experts | MoE 中的专家数量 | None |
| --expert-model-parallel-size | 专家模型并行度 | 1 |
| --moe-ffn-hidden-size | MoE FFN 隐藏层大小 | 密集模型的 FFN 隐藏层大小 |
| --expert-tensor-parallel-size | 专家层张量并行度 | 与 TP 相同（对于细粒度 MoE 模型，建议设置为 1） |
| --moe-layer-freq | MoE 层频率模式 | 1 |

### 路由器参数
| 参数 | 描述 | 默认值 |
|----------|-------------|---------|
| --moe-router-load-balancing-type | 负载均衡类型：aux_loss, sinkhorn, seq_aux_loss, none | aux_loss |
| --moe-router-topk | 每个令牌的专家数量 | 2 |
| --moe-router-score-function | 评分函数：softmax, sigmoid | softmax |
| --moe-router-pre-softmax | 在 top-k 之前进行 softmax | False |
| --moe-router-num-groups | 组限制路由的组数 | None |
| --moe-router-group-topk | 组限制路由中选择的组数 | None |
| --moe-router-enable-expert-bias | 动态的每个专家偏置 | False |
| --moe-router-bias-update-rate | 偏置更新率 | 1e-3 |
| --moe-router-fusion | 启用路由器融合 | False |
| --moe-router-dtype | 路由器精度：fp32, fp64 | None |
| --moe-router-padding-for-fp8 | 为 FP8 对齐进行填充 | False |

### 损失与正则化
| 参数 | 描述 | 默认值 |
|----------|-------------|---------|
| --moe-aux-loss-coeff | 辅助损失系数 | 0.0 |
| --moe-z-loss-coeff | Z 损失系数 | None |
| --moe-input-jitter-eps | 输入抖动 epsilon | None |

### 令牌分发
| 参数 | 描述 | 默认值 |
|----------|-------------|---------|
| --moe-token-dispatcher-type | 分发器类型：allgather, alltoall, flex | allgather |
| --moe-enable-deepep | 启用 DeepEP（与 flex 一起使用） | False |
| --moe-expert-capacity-factor | 容量因子 | None |
| --moe-pad-expert-input-to-capacity | 填充至容量 | False |
| --moe-token-drop-policy | 丢弃策略：probs, position | probs |
| --moe-permute-fusion | 融合置换操作 | False |

### 性能优化
| 参数 | 描述 | 默认值 |
|----------|-------------|---------|
| --moe-grouped-gemm | 使用 GroupedGEMM | False |
| --overlap-moe-expert-parallel-comm | 批处理级别的 EP 通信重叠 | False |
| --delay-wgrad-compute | 分离 dgrad/wgrad 计算 | False |
| --moe-shared-expert-intermediate-size | 共享专家 FFN 大小 | None |
| --moe-shared-expert-overlap | 重叠共享专家 | False |
### 内存与检查点
| 参数 | 描述 | 默认值 |
|----------|-------------|---------|
| --moe-layer-recompute | 重新计算 MoE 层 | False |
| --moe-use-upcycling | 启用循环利用 | False |
| --moe-upcycling-granularity | 循环利用粒度 | 1 |

### 杂项
| 参数 | 描述 | 默认值 |
|----------|-------------|---------|
| --moe-per-layer-logging | 逐层日志记录 | False |
| --moe-router-force-load-balancing | 强制负载均衡（实验性） | False |

## 示例
```bash
#!/bin/bash

# Runs Mixtral 8x7B model on 32 H100/A100 GPUs

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"4"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1
TOKENIZER_MODEL=$2
DATA_PATH=$3

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 8
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --overlap-grad-reduce
    --overlap-param-gather
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --num-layers-per-virtual-pipeline-stage 8
    --sequence-parallel
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```

## 贡献

我们欢迎贡献！请参阅 [CONTRIBUTING.md](https://github.com/NVIDIA/Megatron-LM/blob/main/CONTRIBUTING.md) 了解指南。
## 支持

- GitHub Issues: [报告错误或请求功能](https://github.com/NVIDIA/Megatron-LM/issues)
- 文档: [完整文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)

## 引用

如果您在研究中使用 Megatron-Core MoE，请引用：

```bibtex

@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}

@article{moe-parallel-folding,
    title={MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core}, 
    author={Liu, Dennis and Yan, Zijie and Yao, Xin and Liu, Tong and Korthikanti, Vijay and Wu, Evan and Fan, Shiqing and Deng, Gao and Bai, Hongxiao and Chang, Jianbin and Aithal, Ashwath and Andersch, Michael and Shoeybi, Mohammad and Yao, Jiajie and Zhou, Chandler and Wu, David and Li, Xipeng and Yang, June},
    year={2025},
    journal={arXiv preprint arXiv:2504.14960},
}
```

<!-- nav-links -->

## 相关页面

- [多令牌预测 (MTP)](multi_token_prediction/)
- [多潜在注意力](multi_latent_attention/)
- [设计文档：MoE 路由器重放功能](../../api-guide/router_replay/)
