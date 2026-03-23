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

<!-- nav-links -->

## 关于 Megatron Core

- [概述](get-started/overview/)
- [发布说明](get-started/releasenotes/)

## 快速开始

- [安装](get-started/install/)
- [你的首次训练运行](get-started/quickstart/)

## 基础使用

- [数据准备](user-guide/data-preparation/)
- [训练示例](user-guide/training-examples/)
- [并行策略指南](user-guide/parallelism-guide/)

## 支持的模型

- [支持的模型](models/)

## 高级功能

- [混合专家模型](user-guide/features/moe/)
- [context_parallel 包](user-guide/features/context_parallel/)
- [Megatron FSDP](user-guide/features/custom_fsdp/)
- [分布式优化器](user-guide/features/dist_optimizer/)
- [优化器 CPU 卸载](user-guide/features/optimizer_cpu_offload/)
- [自定义流水线模型并行布局](user-guide/features/pipeline_parallel_layout/)
- [细粒度激活卸载（与 rednote 合作）](user-guide/features/fine_grained_activation_offloading/)
- [Megatron Energon](user-guide/features/megatron_energon/)
- [Megatron RL](user-guide/features/megatron_rl/)
- [Tokenizers](user-guide/features/tokenizers/)

## 开发者指南

- [为 Megatron-LM 做贡献](developer/contribute/)
- [如何提交 PR](developer/submit/)
- [值班概览](developer/oncall/)
- [本地生成文档](developer/generate_docs/)

## API 参考

- [API 指南](api-guide/)

## 资源

- [技术讨论](advanced/)
