<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 多潜在注意力

## 多潜在注意力概述

多潜在注意力（"MLA"）是由 Deepseek 团队引入的一种创新的注意力机制，它通过利用多个潜在空间来提升注意力计算的效率。这种方法对于大型语言模型（LLMs）尤其有益，因为它减轻了与传统注意力机制相关的计算负担。根据 Deepseek-V2 技术报告，MLA 相比多头注意力（MHA）实现了更好的性能，并且需要更小的 KV 缓存。

## 启用多潜在注意力

要在 Megatron-LM 中启用 MLA，请在命令行中设置以下标志：
- `--multi-latent-attention` 以在 MLP 中启用 MLA。
- 设置 `MLATransformerConfig` 来配置 MLA。