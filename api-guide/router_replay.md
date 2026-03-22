<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION 及其许可方保留对本软件、相关文档
   及其任何修改的所有知识产权和专有权利。
   未经 NVIDIA CORPORATION 明示许可协议授权，严禁使用、复制、披露或
   分发本软件及相关文档。
-->

# 设计文档：MoE 路由器重放功能

## 1. 概述

本文档详细描述了在 Megatron-LM Core 中为混合专家（Mixture-of-Experts, MoE）模型实现的“路由器重放”功能。

此功能旨在增强 MoE 模型训练和推理的确定性与可分析性。它使模型能够从预定义的文件加载路由决策，并在前向传播过程中强制使用这些决策，从而绕过实时的路由计算。

## 2. 动机

*   **确定性与可复现性**：在分布式训练中，由于浮点精度等因素，MoE 路由决策可能会出现微小变化。通过重放固定的路由表，可以保证 MoE 计算路径在不同运行中完全一致，这有助于调试和复现实验结果。
*   **性能分析**：路由器自身的计算（例如，逻辑值计算、top-k 选择）会产生开销。在重放模式下，这部分计算可以完全跳过，从而能够更精确地隔离和分析专家层内部的性能瓶颈。
*   **调试辅助**：当模型出现问题时，固定路由决策有助于隔离变量，更容易确定问题是出在路由机制还是专家计算上。

## 3. 设计与架构

设计遵循非侵入式和按需启用的原则，核心思想是仅在用户明确请求时才激活重放逻辑。

*   **核心组件**：
    *   `RouterReplay`（位于 `megatron/core/transformer/moe/router_replay.py`）：一个用于重放 MoE 路由决策的工具类。当通过 `moe_enable_routing_replay` 标志启用时，会为每个 MoE 层的路由器创建一个单独的 `RouterReplay` 实例。每个实例负责加载路由数据，并在前向传播期间为其对应的层提供确定性的路由决策。
    *   `moe_enable_routing_replay`（位于 `megatron/core/transformer/transformer_config.py`）：一个布尔类型的全局配置标志，是启用此功能的唯一入口点。

*   **工作流程**：
    该功能支持不同的模式，例如记录和重放，由 `RouterReplayAction` 控制。

    1.  **启用功能**：用户在模型配置中将 `moe_enable_routing_replay` 设置为 `True`。
    2.  **初始化**：当 `moe_enable_routing_replay` 为 true 时，每个 `TopKRouter` 会创建自己的 `RouterReplay` 实例。
    3.  **模式配置**：用户必须以编程方式在 `RouterReplay` 实例上设置所需的路由器重放操作（例如，`record`、`forward_replay`、`backward_replay`）。
    4.  **执行流程（在一个小批量内）**：
        *   **前向传播**：
            *   对于每个微批次，`topk_routing_with_score_function` 会检查 `router_replay_action`。
            *   **在 `record` 模式下**：捕获并存储动态计算的 `top-k` 专家索引。
            *   **在 `forward_replay` 模式下**：函数从 `target_topk_idx` 中检索预加载的专家索引。这些索引用于前向计算，同时也会被追加到 `replay_backward_list` 中，为后向传播做准备。
        *   **后向传播**：
            *   对于每个微批次（在流水线并行中按相反顺序处理），会再次检查 `router_replay_action`。
            *   **在 `backward_replay` 模式下**：函数通过从 `replay_backward_list` 中弹出对应微批次的专家索引来检索它们。此模式旨在用于训练重计算（例如，激活检查点和流水线重计算），以便在重计算/后向传播期间使用与前向传播相同的路由决策，确保确定性和正确性。
## 4. 实现细节

该实现将重放逻辑与路由器的核心计算清晰地分离开来。

*   **`megatron/core/transformer/transformer_config.py`**:
    *   添加配置选项 `moe_enable_routing_replay: bool = False`。

*   **`megatron/core/transformer/moe/moe_utils.py`**:
    *   引入 `RouterReplay` 类来管理记录和重放单个 MoE 层路由决策的状态。
        *   `target_topk_idx`: 一个属性，在向前重放模式下保存当前微批次（micro-batch）的专家索引。
        *   `recorded_topk_idx`: 一个属性，用于在记录模式下存储计算出的专家索引。
        *   `replay_backward_list`: 一个列表，累积小批次（mini-batch）向前传播过程中使用的 `top-k` 索引。在向后传播期间，该列表以 FIFO（先进先出）顺序被消费，以确保在流水线并行（pipeline parallelism）下的正确性。
        *   `set_target_indices()`: 一个方法，用于将重放索引加载到 `target_topk_idx` 中，供向前传播使用。
        *   `record_indices()`: 一个方法，用于保存计算出的索引。
    *   修改了 `topk_routing_with_score_function` 以包含核心逻辑。它检查 `router_replay` 实例上的 `router_replay_action`，并相应地执行以下操作之一：计算并记录索引、从 `target_topk_idx` 重放索引（用于向前传播）、从 `replay_backward_list` 重放索引（用于向后传播），或者回退到默认的动态路由。

### 训练重计算使用

- 在向前重放期间，`set_target_indices()` 会准备 `replay_backward_list`，以便每个微批次的索引都可用于重计算。
- 在重计算/向后传播期间，将动作设置为 `REPLAY_BACKWARD`，以便以 FIFO 顺序消费索引，从而与向前传播序列保持一致。

## 5. 使用指南

1.  **启用与实例化**
    - 在构建模型时，为每个 MoE 路由器层创建一个 `RouterReplay` 实例。
    - 可选地使用全局辅助函数来跨所有层设置/清除动作。
2.  **记录路由决策**
    - 设置动作：`RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)`。
    - 运行模型；通过 `RouterReplay.get_recorded_data()` 获取每层的索引并持久化保存。
3.  **向前重放**
    - 加载索引并分发：`RouterReplay.set_replay_data(list_of_tensors)`。
    - 设置动作：`RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)`。
    - 运行模型；动态 top‑k 被绕过，使用目标索引。
4.  **向后重放**
    - 对于训练重计算（激活检查点或流水线重计算），在重计算期间设置动作：`REPLAY_BACKWARD`。
    - 每个微批次的索引按照 FIFO 顺序从 `replay_backward_list` 中消费。
5.  **清理**
    - 使用 `RouterReplay.clear_global_indices()`、`RouterReplay.clear_global_router_replay_action()` 和 `RouterReplay.clear_global_router_replay_instances()` 来恢复默认行为并防止内存泄漏。
### 使用 `topk_routing_with_score_function` 快速上手

```python
import torch
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

rr = RouterReplay()

# 记录模式
RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
logits = torch.randn(8, 16)
probs_rec, routing_map_rec = topk_routing_with_score_function(
    logits=logits, topk=2, use_pre_softmax=False, score_function="softmax", router_replay=rr,
)
recorded = rr.get_recorded_indices()
torch.save(recorded, "/tmp/replay.pt")

# 前向重放模式
rr.clear_router_replay_action()
rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
target = torch.load("/tmp/replay.pt")
rr.set_target_indices(target)
probs_rep, routing_map_rep = topk_routing_with_score_function(
    logits=logits, topk=2, use_pre_softmax=False, score_function="softmax", router_replay=rr,
)

RouterReplay.clear_global_router_replay_action()
RouterReplay.clear_global_indices()
RouterReplay.clear_global_router_replay_instances()
```

## 6. 最小化演示

以下是一个展示如何使用 RouterReplay 进行记录和重放的最小化代码示例：

```python
import torch
import torch.distributed as dist
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


# 初始化分布式训练
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

# 创建一个启用了 RouterReplay 的 transformer 配置
config = TransformerConfig(
    num_experts=8,
    expert_model_parallel_size=1,
    num_top_k=2,
    moe_enable_routing_replay=True
)

# 创建一个 TopKRouter 实例
router = TopKRouter(config)

# 生成示例输入 (batch_size, sequence_length, hidden_size)
logits = torch.randn(16, 32, 8).to(torch.cuda.current_device())

# -----------------
# 1. 记录模式
# -----------------
print("=== 记录模式 ===")
# 设置全局路由器重放动作为 RECORD
RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

# 执行路由
routing_output = router.forward(logits)
print(f"记录的 top-k 索引形状: {routing_output.top_k_idx.shape}")

# -----------------
# 2. 前向重放模式
# -----------------
print("\n=== 前向重放模式 ===")
# 将记录的索引保存到文件
torch.save(routing_output.top_k_idx, "/tmp/replay.pt")

# 从文件加载索引并设置为重放目标
replay_indices = torch.load("/tmp/replay.pt")
for router_instance in RouterReplay.global_router_replay_instances:
    router_instance.target_topk_idx = replay_indices

# 设置全局路由器重放动作为 REPLAY_FORWARD
RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

# 再次执行路由 - 这将使用重放的索引
replay_routing_output = router.forward(logits)
print(f"重放的 top-k 索引形状: {replay_routing_output.top_k_idx.shape}")
print(f"索引是否相同? {torch.equal(routing_output.top_k_idx, replay_routing_output.top_k_idx)}")


# 清理
RouterReplay.clear_global_router_replay_action()
RouterReplay.clear_global_indices()
RouterReplay.clear_global_router_replay_instances()
if dist.is_initialized():
    dist.destroy_process_group()
```