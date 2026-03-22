<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 多模态模型

Megatron Core 支持结合语言、视觉、音频和其他模态的多模态模型，以实现全面的多模态理解。

## MIMO：多模态输入/输出框架

**MIMO（多模态输入/输出模型）** 是 Megatron Core 中的一个实验性框架，支持包括视觉、音频和文本在内的任意模态组合。MIMO 为构建自定义多模态模型提供了一个灵活的架构。

> **注意**：MIMO 是实验性的，正在积极开发中。API 可能在未来的版本中发生变化。

**主要特性：**
- 任意模态组合（视觉、音频、文本等）
- 针对不同输入模态的灵活编码器架构
- 跨模态的统一嵌入空间
- 支持视觉-语言和音频-视觉-语言模型

请参阅 [examples/mimo](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mimo) 获取训练脚本和示例。

## 视觉-语言模型

| 模型 | 描述 | 视觉编码器 | 语言模型 |
|-------|-------------|----------------|----------------|
| **LLaVA** | 视觉指令微调 | CLIP ViT-L/14 | Mistral-7B / LLaMA |
| **NVLM** | NVIDIA 视觉-语言模型 | CLIP / 自定义 ViT | 基于 LLaMA |
| **LLaMA 3.1 Nemotron Nano VL** | 高效多模态模型 | Vision Transformer | LLaMA 3.1 8B |

## 视觉编码器

| 模型 | 描述 | 关键特性 |
|-------|-------------|--------------|
| **CLIP ViT** | OpenAI 的 CLIP Vision Transformer | 图文对齐，多种尺度（L/14@336px） |
| **RADIO** | 分辨率无关的动态图像优化 | 灵活的分辨率处理，高效的视觉编码 |

## 扩散模型

关于多模态扩散模型（图像生成、文生图等），请参阅 [NeMo Diffusion Models](https://github.com/NVIDIA-NeMo/NeMo/tree/main/nemo/collections/diffusion)。NeMo 提供了生产就绪的实现，包括：
- Stable Diffusion 变体
- 文本到图像生成
- 图像到图像转换
- ControlNet 和其他条件机制

## 多模态特性

- **图文对齐**：在图像-标题对上进行预训练
- **视觉指令微调**：在遵循指令的数据集上进行微调
- **灵活的视觉编码器**：支持不同的 ViT 架构和分辨率
- **组合式检查点**：结合视觉和语言模型的统一检查点
- **高效训练**：对视觉和语言组件均提供完整的并行支持（TP、PP、DP）

## 示例脚本

多模态训练示例可在以下目录中找到：
**MIMO 框架：**
- `examples/mimo/` - 支持视觉-语言和音频-视觉-语言模型的多模态输入/输出训练

**特定多模态模型：**
- `examples/multimodal/` - 使用 Mistral + CLIP 的 LLaVA 风格训练
- `examples/multimodal/nvlm/` - NVLM 训练脚本
- `examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/` - Nemotron VL 训练
- `examples/multimodal/radio/` - RADIO 视觉编码器集成