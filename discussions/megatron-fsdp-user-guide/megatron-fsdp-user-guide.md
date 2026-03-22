---
orphan: true
---

<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Megatron-FSDP 用户指南

## 目录

- [Megatron-FSDP 快速入门](#megatron-fsdp-快速入门)
- [从 3D 并行到 Megatron-FSDP 的检查点转换](#从-3d-并行到-megatron-fsdp-的检查点转换)

## Megatron-FSDP 快速入门

我们推荐使用最新的 [NVIDIA NeMo 框架容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)，它提供了经过测试的软件栈和优化的性能。

供您参考，我们提供了一个用于 DeepSeek-V3 的示例启动脚本：[`sbatch_mfsdp_deepseek_v3.sh`](./example-scripts/sbatch_mfsdp_deepseek_v3.sh)。

### 必需配置

要启用 Megatron-FSDP，请在您的训练脚本中添加以下必需标志：

```bash
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--no-gradient-accumulation-fusion
--use-distributed-optimizer
--ckpt-format fsdp_dtensor
```

### 推荐配置

我们还建议添加以下配置以进一步提升性能：

```bash
unset CUDA_DEVICE_MAX_CONNECTIONS
```
```bash
--calculate-per-token-loss
--init-model-with-meta-device
--grad-reduce-in-bf16
--fsdp-double-buffer
--use-nccl-ub
```

💡 **这些配置的详细说明如下。**

#### 1. 禁用 `CUDA_DEVICE_MAX_CONNECTIONS`

为确保 FSDP 通信和计算的完全并行化，请禁用 `CUDA_DEVICE_MAX_CONNECTIONS` 环境变量。此步骤可避免 CUDA 流中潜在的气泡。（但这可能会在一定程度上减慢 TP 和 CP。）

#### 2. 添加 `--calculate-per-token-loss`

对于梯度分片模式优化，请在您的训练脚本中包含 `--calculate-per-token-loss` 标志。这通过减少梯度缩放的频率来提高性能，梯度缩放也是 SM 资源的一个相当大的消耗。

#### 3. 添加 `--init-model-with-meta-device`

允许使用元设备（meta device）初始化模型，然后通过 `Module.reset_parameters` API 逐层初始化分布式模型权重缓冲区，便于初始化极大的模型。

#### 4. 添加 `--grad-reduce-in-bf16`

启用 BF16 精度而非 FP32 精度的梯度规约，减少通信量并加速反向传播。

#### 5. 添加 `--fsdp-double-buffer`

为 `MegatronFSDP` 通信中所需的临时定义内存使用持久分配的双缓冲区。虽然拥有持久的双缓冲区可能会增加峰值 VRAM 使用率，但这是为 `MegatronFSDP` 注册 NCCL 用户缓冲区（`nccl_ub=True`）所必需的。目前，这仅支持简单的重复模型结构，例如 GPT。
- **仅在使用 Megatron-LM 时有效。**
- 默认为 `False`。当启用 `nccl_ub` 时，会自动覆盖为 `True`。

#### 6. 添加 `--use-nccl-ub`

为参数和梯度缓冲区分配并[注册 NCCL 用户缓冲区](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#)。此选项启用一种 SM 高效的 NCCL 算法，可以提高重叠计算的性能。如果 FSDP 通信同时包含 NVL 和 IB 域，此标志与 [SHARP](https://docs.nvidia.com/networking/display/sharpv3130) 一起使用时效果会更显著。启用此选项将导致额外的内存开销，因为需要启用 `fsdp_double_buffer` 选项。

- **仅在使用 Megatron-LM 时有效。**
- 默认为 `False`。
- 默认情况下，如果可用，我们会尝试使用 NCCL 窗口（对称）注册。否则，它将回退到传统的本地注册。
- **与 PyTorch 的可分段分配器不兼容：** 使用 `--use-nccl-ub` 时，不要设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，因为这会导致与 `torch.cuda.MemPool` API 的兼容性问题，从而引发运行时错误。

## 从 3D 并行检查点转换到 Megatron-FSDP

Megatron-FSDP 引入了 `fsdp_dtensor`，这是一种基于 DTensor 的分布式检查点格式，作为其标准。为了帮助您顺利地从 3D 并行过渡到 Megatron-FSDP，我们提供了一个脚本，用于将检查点从 `torch_dist` 格式转换为 `fsdp_dtensor` 格式。以 DeepSeek-V3 为例，详细的转换过程如下所述。

### 步骤 1：使用 `param_to_param_group_map` 生成 3D 并行检查点

运行您的 3D 并行 + EP 训练脚本，生成一个 `torch_dist` 检查点以及一个包含 `param_to_param_group_map` 文件的目录。在您的训练脚本中添加以下标志：

```bash
--dump-param-to-param-group-map /path/to/param_to_param_group_map
```

如果您已经有一个 `torch_dist` 检查点，只需指定 `--dump-param-to-param-group-map /path/to/param_to_param_group_map` 标志并运行一个非常短的实验——这将创建您需要的 `param_to_param_group_map`，而无需完整的预训练。

### 步骤 2：将 `param_to_param_group_map` 导出为 JSON 文件

通过运行以下命令，将 `param_to_param_group_map` 转换为 JSON 文件以便于处理：

```bash
python tools/checkpoint/checkpoint_inspector.py print-torch-dcp-in-json /path/to/param_to_param_group_map
```

这将在 `/path/to/param_to_param_group_map` 目录中创建一个 `param_to_param_group_map.json` 文件。

### 步骤 3：将检查点从 `torch_dist` 转换为 `fsdp_dtensor`

使用参数到 `param_to_param_group_map` 的 JSON 文件，将您的 `torch_dist` 检查点转换为 `fsdp_dtensor` 格式：

```bash
torchrun --nproc_per_node=8 --nnodes=1 \
    tools/checkpoint/checkpoint_inspector.py \
    convert-torch-dist-to-fsdp-dtensor --swiglu \
    /path/to/input_torch_dist_checkpoint \
    /path/to/output_fsdp_dtensor_checkpoint \
    --param-to-param-group-map-json /path/to/param_to_param_group_map.json
```
**注意：** 对于多节点转换任务，请参考示例脚本：[`sbatch_checkpoint_convert.sh`](./example-scripts/sbatch_checkpoint_convert.sh)。

### 步骤 4：启动 Megatron-FSDP 训练

使用转换后的 `fsdp_dtensor` 检查点启动您的 Megatron-FSDP 训练任务。