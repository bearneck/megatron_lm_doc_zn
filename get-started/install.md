<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 安装

## 系统要求

### 硬件

- **推荐**: NVIDIA Turing 架构或更高版本
- **FP8 支持**: 需要 NVIDIA Hopper、Ada 或 Blackwell GPU

### 软件

- **Python**: >= 3.10 (推荐 3.12)
- **PyTorch**: >= 2.6.0
- **CUDA Toolkit**: 最新的稳定版本


## 前置条件

安装 [uv](https://docs.astral.sh/uv/)，一个快速的 Python 包安装器：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


## 选项 A: Pip 安装 (推荐)

从 PyPI 安装最新的稳定版本：

```bash
uv pip install megatron-core
```

要包含可选的训练依赖项 (Weights & Biases, SentencePiece, HF Transformers)：

```bash
uv pip install "megatron-core[training]"
```

要包含所有额外项，包括 [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)：

```bash
uv pip install --group build
uv pip install --no-build-isolation "megatron-core[training,dev]"
```

```{note}
`--no-build-isolation` 要求构建依赖项已预先安装在环境中。需要 `torch` 是因为几个 `[dev]` 包 (`mamba-ssm`, `nv-grouped-gemm`, `transformer-engine`) 在构建时导入它以编译 CUDA 内核。根据您的硬件情况，预计此步骤需要 **20 分钟以上**。如果您更喜欢预构建的二进制文件，[NGC 容器](#option-c-ngc-container) 已预编译了这些内容。
```

```{warning}
从源代码构建会消耗大量内存。默认情况下，构建过程会为每个 CPU 核心运行一个编译器作业，这可能会导致在具有许多核心的机器上出现内存不足故障。要限制并行编译作业数，请在安装前设置 `MAX_JOBS` 环境变量（例如 `MAX_JOBS=4`）。
```

```{tip}
对于一组更轻量的开发依赖项（不含 Transformer Engine 和 ModelOpt），请使用 `[lts]` 而不是 `[dev]`：`uv pip install --no-build-isolation "megatron-core[training,lts]"`。`[lts]` 和 `[dev]` 额外项是互斥的。
```

要克隆仓库以获取示例：

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
```


## 选项 B: 从源代码安装

用于开发或运行最新的未发布代码：

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

要安装所有开发依赖项（包括 Transformer Engine，需要预先安装构建依赖项）：

```bash
uv pip install --group build
uv pip install --no-build-isolation -e ".[training,dev]"
```

```{tip}
如果构建过程内存不足，请使用 `MAX_JOBS=4 uv pip install --no-build-isolation -e ".[training,dev]"` 来限制并行编译作业数。
```
## 选项 C：NGC 容器

如需使用预配置环境（已预装 PyTorch、CUDA、cuDNN、NCCL、Transformer Engine 等所有依赖项），请使用 [PyTorch NGC 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)。

我们建议使用**上个月**的 NGC 容器，而非最新版本，以确保与当前 Megatron Core 版本和测试矩阵的兼容性。

```bash
docker run --gpus all -it --rm \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:26.01-py3
```

```{note}
NGC PyTorch 容器通过 `PIP_CONSTRAINT` 全局约束 Python 环境。上述命令中的 `-e PIP_CONSTRAINT=` 标志会取消此约束，以便正确安装 Megatron Core 及其依赖项。
```

然后在容器内安装 Megatron Core（NGC 镜像中已包含 torch）：

```bash
pip install uv
uv pip install --no-build-isolation "megatron-core[training,dev]"
```

现在您已准备好运行训练。后续步骤请参阅[您的首次训练运行](quickstart.md)。