<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION 及其许可方保留对本软件、相关文档
   及其任何修改的所有知识产权和专有权利。
   未经 NVIDIA CORPORATION 明确许可协议授权，严禁任何使用、
   复制、披露或分发本软件及相关文档的行为。
-->

# dist_checkpointing 包

一个用于保存和加载分布式检查点的库。
Megatron Core 中的*分布式检查点*使用 ``torch_dist`` 格式，
这是一种基于 PyTorch 原生检查点功能构建的自定义检查点机制。

分布式检查点的一个关键特性是，在一个并行配置（张量并行、流水线并行或数据并行）下保存的检查点，
可以在不同的并行配置下加载。这使得模型能够在异构训练设置中进行灵活的缩放和重新分片。

使用该库需要通过 *mapping* 和 *optimizer* 模块中的函数定义分片的状态字典（state_dict）。
这些状态字典可以使用 *strategies* 模块中的策略，通过 *serialization* 模块进行保存或加载。

## 安全加载检查点

自 **PyTorch 2.6** 起，`torch.load` 的默认行为是 `weights_only=True`。
这确保只加载张量和允许列表中的类，降低了任意代码执行的风险。

如果遇到如下错误：

```bash
WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace was not an allowed global by default.
```

可以通过在脚本中明确将缺失的类添加到安全全局列表中来解决：

```python
import torch, argparse

torch.serialization.add_safe_globals([argparse.Namespace])
```

## 检查点分布式优化器

### 检查点兼容性与优化器状态格式

从 **mcore v0.14** 开始，`dist_checkpointing` 中移除了 `flattened_range` 属性。因此：

-   使用 mcore 版本 <= 0.14 保存的优化器状态无法再直接加载。由于所需的分片元数据不再可用，不支持加载这些旧的优化器状态。如果需要从旧检查点继续训练，请参考下面描述的解决方法。
-   旧检查点中的模型权重保持完全兼容。无需额外步骤——早期版本创建的检查点中的模型权重会自动加载；只需添加 `--no-load-optim` 标志即可。

### 解决方法：使用 ToT MCore 加载旧的优化器状态

**步骤 1：使用 mcore v0.15.0 转换旧检查点**

使用 mcore v0.15.0 运行一个虚拟训练任务，以新的优化器状态格式重新保存检查点。

```bash
MODEL_TRAIN_PARAMS=(
    # 在此处定义模型架构和训练参数
)
OLD_CKPT=/workspace/mcore_ckpt_old
CONVERTED_CKPT=/workspace/mcore_ckpt_0.15.0

torchrun --nproc_per_node=8 /opt/megatron-lm/pretrain_gpt.py \
   --save-interval 1 \
   --eval-interval 1 \
   --exit-interval 1 \
   --eval-iters 1 \
   --use-distributed-optimizer \
   --save ${CONVERTED_CKPT} \
   --load ${OLD_CKPT} \
   --ckpt-format torch_dist \
   "${MODEL_TRAIN_PARAMS[@]}"
```
**步骤 2：使用 ToT MCore 加载转换后的检查点**

将转换后的检查点用作使用 ToT MCore 进行继续训练的输入。

```bash
MODEL_TRAIN_PARAMS=(
    # 在此处定义模型架构和训练参数
)
NEW_CKPT=/workspace/mcore_ckpt_new
CONVERTED_CKPT=/workspace/mcore_ckpt_0.15.0

torchrun --nproc_per_node=8 /opt/megatron-lm/pretrain_gpt.py \
   --use-distributed-optimizer \
   --save ${NEW_CKPT} \
   --load ${CONVERTED_CKPT} \
   --ckpt-format torch_dist \
   "${MODEL_TRAIN_PARAMS[@]}"
```

完成此步骤后，即可使用 ToT MCore 正常进行训练，并完全支持优化器状态加载。

## 分布式优化器检查点格式

分布式优化器的重构引入了**两种检查点格式**：

- dp_reshardable（默认）
   - 保存/加载性能快。
   - 不可重分片 — 使用此格式时无法更改模型并行度。
   - 当不需要更改模型并行度时，推荐用于常规训练。
- fully_reshardable
   - 完全可重分片 — 支持任意更改模型并行度。
   - 比 dp_reshardable 慢。
   - 通过 ``--dist-ckpt-optim-fully-reshardable`` 标志启用。

### 更改模型并行度的工作流程

您可以组合使用格式以优化灵活性和性能：

   1. 使用 ``dp_reshardable``（默认）进行训练以获得更快的检查点保存速度。
   2. 当需要更改模型并行度时：

      - 停止训练。
      - 更改训练配置中的模型并行度。
      - 使用 ``--dist-ckpt-optim-fully-reshardable`` 恢复训练。

   3. 在新的模型并行配置下至少保存一个检查点。
   4. （可选）为了在更新模型并行度后继续训练并获得更好的检查点性能，可以停止训练并通过移除 ``--dist-ckpt-optim-fully-reshardable`` 切换回 ``dp_reshardable`` 格式。

## 子包

```{toctree}
:maxdepth: 4

dist_checkpointing.strategies
```