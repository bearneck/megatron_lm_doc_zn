<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 分布式优化器

分布式优化器的动机是通过将优化器状态均匀分布在数据并行（Data Parallel, DP）的各个 rank 上（https://arxiv.org/abs/1910.02054）来节省内存，这与在数据并行 rank 间复制优化器状态的朴素方法形成对比。

理论上的内存节省量取决于模型参数的数据类型（`param_dtype`）与跨数据并行副本累积的主梯度数据类型（`grad_dtype`）的组合。我们始终使用 `fp32` 主参数进行优化器步骤。在当前实现中，每个参数的理论字节数如下（其中 d 是数据并行规模）：

|        | 非分布式优化器 | 分布式优化器 |
| ------ | ------ | ------ |
| `fp16` 参数, `fp16` 梯度 | 20 | 4 + 16/d |
| `bf16` 参数, `fp32` 梯度    | 18 | 6 + 12/d |
| `fp32` 参数, `fp32` 梯度       | 16 | 8 + 8/d  |

我们实现的分布式优化器使用连续的缓冲区来存储参数和主梯度；模型梯度一旦完全计算出来，就会被复制到主梯度中。

下图说明了分布式优化器的分片方案，以及分布式优化器参数更新的关键步骤：

## 数据流

![数据流](../../images/distrib_optimizer/data_flow.png)

## 分片方案

![分片方案](../../images/distrib_optimizer/sharding_scheme.png)

## 关键步骤

_(注意：使用上图说明，假设模型权重为 `bf16`，反向传播计算出的模型梯度为 `bf16`，而用于优化器步骤的主梯度为 `fp32`；我们始终使用 `fp32` 主权重进行优化器步骤)_

- 反向传播完成（梯度缓冲区持有 16 个 `fp32` 梯度元素）。
- 在每个 DP rank 上调用 reduce-scatter 操作。
- 每个 DP rank 现在在梯度缓冲区中拥有 4 个已完全规约的元素（剩余的 12 个元素是无效数据）。
  - DP rank 0 拥有元素 [0:4] 的梯度值。
  - DP rank 1 拥有元素 [4:8] 的梯度值。
  - DP rank 2 拥有元素 [8:12] 的梯度值。
  - DP rank 3 拥有元素 [12:16] 的梯度值。
- 调用 Optimizer.step()。
- 每个 DP rank 将其 4 个 `fp32` 主参数元素复制到对应的 `bf16` 参数缓冲区中（每个元素从 fp32 转换为 fp16）。
- 在每个 DP rank 上调用 all-gather 操作。
- 参数缓冲区现在包含了全部 16 个已完全更新的 `bf16` 模型参数元素。PyTorch 模块中的参数已经指向此参数缓冲区中的适当位置，因此在前向传播可以在 all-gather 完成后立即运行。
- 此时，梯度缓冲区也已准备好在下一次迭代前被清零。