<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION 及其许可方保留对本软件、相关文档
   及其任何修改的所有知识产权和专有权利。未经 NVIDIA CORPORATION
   明确许可协议授权，严禁任何使用、复制、披露或分发本软件及相关文档的行为。
-->

# distributed 包

此包包含各种实用工具，用于在优化器步骤之前完成每个 rank 上模型权重的梯度计算。这包括一个分布式数据并行包装器，用于跨数据并行副本进行 all-reduce 或 reduce-scatter 梯度操作，以及一个 `finalize_model_grads` 方法，用于跨不同并行模式同步梯度（例如，不同流水线阶段上的 'tied' 层，或由于专家并行导致在不同 rank 上的 MoE 专家梯度）。