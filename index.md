<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION 及其许可方保留对本软件、相关文档
   及其任何修改的所有知识产权和专有权利。
   未经 NVIDIA CORPORATION 明确许可协议授权，严禁任何使用、
   复制、披露或分发本软件及相关文档的行为。
-->

# Megatron Core 用户指南

**Megatron Core** 是一个针对 GPU 优化的库，用于大规模训练大型语言模型。它提供了模块化、可组合的构建块，用于创建具有最先进的并行策略和性能优化的自定义训练框架。

Megatron Core 为构建大规模 Transformer 训练系统提供了一个灵活、可重用的基础。**Megatron-LM** 作为一个参考实现，展示了如何使用 Megatron Core 组件在分布式 GPU 集群上训练具有数十亿到数万亿参数的模型。

## 主要特性

*   可组合的 Transformer 构建块（注意力机制、MLP）
*   先进的并行策略（TP、PP、DP、EP、CP）
*   流水线调度和分布式优化器
*   混合精度支持（FP16、BF16、FP8）
*   GPU 优化的内核和内存管理
*   高性能数据加载器和数据集工具
*   模型架构（LLaMA、Qwen、DeepSeek、GPT、Mamba）

```{toctree}
:maxdepth: 2
:hidden:
:caption: 关于 Megatron Core

get-started/overview
get-started/releasenotes
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 快速开始

get-started/install
get-started/quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 基础使用

user-guide/data-preparation
user-guide/training-examples
user-guide/parallelism-guide
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 支持的模型

models/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 高级特性

user-guide/features/moe
user-guide/features/context_parallel
user-guide/features/custom_fsdp
user-guide/features/dist_optimizer
user-guide/features/optimizer_cpu_offload
user-guide/features/pipeline_parallel_layout
user-guide/features/fine_grained_activation_offloading
user-guide/features/megatron_energon
user-guide/features/megatron_rl
user-guide/features/tokenizers
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 开发者指南

developer/contribute
developer/submit
developer/oncall
developer/generate_docs
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API 参考

api-guide/index
apidocs/index.rst
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 资源

advanced/index
```