<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION 及其许可方保留对本软件、相关文档
   及其任何修改的所有知识产权和专有权利。未经 NVIDIA CORPORATION
   明确许可协议授权，严禁任何使用、复制、披露或分发本软件及相关文档的行为。
-->

# Megatron RL

用于大规模后训练大型语言模型的强化学习库。

## 概述

[**Megatron RL**](https://github.com/NVIDIA/Megatron-LM/tree/dev/megatron/rl) 为 Megatron-LM 增加了原生强化学习能力，用于对基础模型进行大规模的基于 RL 的后训练。

> **注意**：Megatron RL 正在积极开发中，主要面向在现代 NVIDIA 硬件上探索 RL 后训练的研究团队。对于生产部署，请使用 [**NeMo RL**](https://github.com/NVIDIA-NeMo/RL)。

## 主要特性

- **解耦设计** - 智能体/环境逻辑与 RL 实现之间清晰分离
- **灵活推理** - 支持 Megatron、OpenAI 和 HuggingFace 推理后端
- **训练器/评估器** - 管理经验轨迹生成并与推理系统协调
- **Megatron 集成** - 与 Megatron Core 推理系统原生集成

## 架构

### 组件

**智能体与环境**
- 接受推理句柄
- 返回带有奖励的经验轨迹
- 实现自定义 RL 逻辑

**训练器/评估器**
- 控制轨迹生成
- 与推理系统协调
- 管理训练循环

**推理接口**
- 提供 `.generate(prompt, **generation_args)` 端点
- 支持多种后端（Megatron、OpenAI、HuggingFace）

## 使用场景

- RLHF（基于人类反馈的强化学习）
- 基于自定义奖励的微调
- 针对特定任务的策略优化
- RL 后训练技术研究

## 资源

- **[Megatron RL GitHub](https://github.com/NVIDIA/Megatron-LM/tree/dev/megatron/rl)** - 源代码和文档
- **[Megatron Core Inference](../../api-guide/core/transformer.md)** - 原生推理集成