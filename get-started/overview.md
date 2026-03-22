<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 概述

Megatron-Core 和 Megatron-LM 是开源工具，通常结合使用以在多个 GPU 上大规模训练 LLM。Megatron-Core 扩展了 Megatron-LM 的能力。Megatron Bridge 将 Megatron-Core 和 Megatron-LM 连接到其他流行的训练模型，例如 Hugging Face。

## Megatron Core

NVIDIA Megatron Core 是一个用于高效大规模生成式 AI 训练的基础构建块库。它可用于在数千个 GPU 上以无与伦比的速度大规模训练模型。它提供了一套广泛的工具，用于多模态和语音 AI。它扩展了 Megatron LM 的能力。

Megatron-Core 包含 GPU 优化的技术，具有先进的并行策略、FP8 训练等优化，以及对最新 LLM、MoE 和多模态架构的支持。它将这些技术抽象为可组合和模块化的 API。

Megatron-Core 兼容所有 NVIDIA Tensor Core GPU 和流行的 LLM 架构，如 GPT、BERT、T5 和 RETRO。

**可组合库**，包含用于自定义训练框架的 GPU 优化构建块。

**最适合：**

- **框架开发者**，基于模块化和优化的组件进行构建
- **研究团队**，需要自定义训练循环、优化器或数据管道
- **机器学习工程师**，需要容错的训练管道

**您将获得：**

- 可组合的 Transformer 构建块（注意力、MLP）
- 先进的并行策略（TP、PP、DP、EP、CP）
- 流水线调度和分布式优化器
- 混合精度支持（FP16、BF16、FP8）
- GPU 优化的内核和内存管理
- 高性能数据加载器和数据集实用工具
- 模型架构（LLaMA、Qwen、GPT、Mixtral、Mamba）

## Megatron-LM

Megatron-LM 是一个参考实现，包含一个轻量级的大规模 LLM 训练框架。它提供了一个可定制的原生 PyTorch 训练循环，抽象层更少。它旨在在现实的内存和计算限制下，将 Transformer 模型扩展到数十亿和万亿参数规模。**它是探索 Megatron-Core 的一个直接入口点。**

它使用先进的并行化技术，包括模型并行（张量并行和流水线并行），使得具有数十亿参数的模型能够适应并在大型 GPU 集群上进行训练。它在大规模 NLP 任务中实现了突破。它将模型计算拆分到多个 GPU 上，克服了单 GPU 内存限制，从而能够训练像 GPT 风格 Transformer 这样的巨大模型。

**参考实现**，包含 Megatron Core 以及训练模型所需的一切。
**最佳适用场景：**

- **大规模训练最先进的基础模型**，在最新的 NVIDIA 硬件上实现尖端性能
- **研究团队**探索新架构和训练技术
- **学习分布式训练**概念和最佳实践
- **快速实验**经过验证的模型配置

**您将获得：**

- 针对 GPT、LLaMA、DeepSeek、Qwen 等模型的预配置训练脚本
- 从数据准备到评估的端到端示例
- 专注于研究的工具和实用程序



## Megatron Bridge

Megatron Bridge 为基于 Megatron Core 基础模型架构构建的模型提供了开箱即用的桥接器和训练方案。

Megatron Bridge 提供了一个健壮的、感知并行性的路径来转换模型和检查点。这个双向转换器执行即时、感知模型并行、逐参数转换以及完全的内存加载。

在训练或修改 Megatron 模型后，您可以再次转换它以进行部署或共享。

[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)



## 生态系统库

**Megatron Core 使用的库：**

- **[Megatron Energon](https://github.com/NVIDIA/Megatron-Energon)** - 多模态数据加载器（文本、图像、视频、音频），支持分布式加载和数据集混合
- **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine)** - 优化的内核和 FP8 混合精度支持
- **[Resiliency Extension (NVRx)](https://github.com/NVIDIA/nvidia-resiliency-ext)** - 具有故障检测和恢复功能的容错训练

**使用 Megatron Core 的库：**

- **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - 训练库，提供双向 Hugging Face ↔ Megatron 检查点转换、灵活的训练循环以及生产就绪的方案
- **[NeMo RL](https://github.com/NVIDIA-NeMo/RL)** - 可扩展的工具包，用于通过 RLHF、DPO 和其他后训练方法进行高效的强化学习
- **[NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)** - 企业级框架，支持云原生并提供端到端示例
- **[Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer)** - 模型优化工具包，用于量化、剪枝、蒸馏、推测解码等。请在 [examples/post_training/modelopt](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt) 中查看端到端示例。

**兼容于：** [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [Colossal-AI](https://github.com/hpcaitech/ColossalAI), [DeepSpeed](https://github.com/microsoft/DeepSpeed)