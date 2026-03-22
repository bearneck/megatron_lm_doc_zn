<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# distributed 包

此包包含多种实用工具，用于在优化器步骤之前，在每个 rank 上完成模型权重梯度的最终处理。这包括一个分布式数据并行包装器，用于跨数据并行副本进行 all-reduce 或 reduce-scatter 梯度操作，以及一个 `finalize_model_grads` 方法，用于跨不同并行模式（例如，不同流水线阶段上的 'tied' 层，或由于专家并行而分布在不同 rank 上的 MoE 专家梯度）同步梯度。