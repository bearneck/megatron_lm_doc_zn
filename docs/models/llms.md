<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION 及其许可方保留对本软件、相关文档
   及其任何修改的所有知识产权和专有权利。
   未经 NVIDIA CORPORATION 明确许可协议授权，严禁任何使用、
   复制、披露或分发本软件及相关文档的行为。
-->

# 语言模型

Megatron Core 支持以下用于大规模训练的语言模型架构。

## 转换 HuggingFace 模型

使用 [**Megatron Bridge**](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 将 HuggingFace 模型转换为 Megatron 格式。Megatron Bridge 是官方的独立转换器，支持广泛的模型列表，包括 LLaMA、Mistral、Mixtral、Qwen、DeepSeek、Gemma、Phi、Nemotron 等。

有关完整且最新的列表，请参阅 [Megatron Bridge 支持的模型列表](https://github.com/NVIDIA-NeMo/Megatron-Bridge?tab=readme-ov-file#supported-models)。

## 仅解码器模型

| 模型 | 描述 | 关键特性 |
|-------|-------------|--------------|
| **GPT** | 生成式预训练 Transformer | 标准自回归语言模型，基础架构 |
| **LLaMA** | Meta 的 LLaMA 系列 | 采用 RoPE、SwiGLU、RMSNorm 的高效架构 |
| **Mistral** | Mistral AI 模型 | 滑动窗口注意力，高效推理 |
| **Mixtral** | 稀疏混合专家模型 | 8x7B MoE 架构，用于高效扩展 |
| **Qwen** | 阿里的 Qwen 系列 | HuggingFace 集成，多语言支持 |
| **Mamba** | 状态空间模型 | 次二次序列长度扩展，高效长上下文处理 |

## 仅编码器模型

| 模型 | 描述 | 关键特性 |
|-------|-------------|--------------|
| **BERT** | 双向编码器表示 | 掩码语言建模，分类任务 |

## 编码器-解码器模型

| 模型 | 描述 | 关键特性 |
|-------|-------------|--------------|
| **T5** | 文本到文本迁移 Transformer | 统一的文本到文本框架，序列到序列 |

## 示例脚本

这些模型的训练示例可在 `examples/` 目录中找到：
- `examples/gpt3/` - GPT-3 训练脚本
- `examples/llama/` - LLaMA 训练脚本
- `examples/mixtral/` - Mixtral MoE 训练
- `examples/mamba/` - Mamba 训练脚本
- `examples/bert/` - BERT 训练脚本
- `examples/t5/` - T5 训练脚本

## 模型实现

所有语言模型均使用 Megatron Core 的可组合 Transformer 块构建，支持：
- 灵活的并行策略（张量并行 TP、流水线并行 PP、数据并行 DP、专家并行 EP、序列并行 CP）
- 混合精度训练（FP16、BF16、FP8）
- 分布式检查点
- 高效内存管理