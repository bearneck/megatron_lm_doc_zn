<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Megatron Energon

用于大规模高效加载文本、图像、视频和音频的高级多模态数据加载器。

## 概述

[**Megatron Energon**](https://github.com/NVIDIA/Megatron-Energon) 专为大规模多模态训练而构建，具有以下特点：

- **多模态支持** - 文本、图像、视频、音频
- **分布式加载** - 针对多节点训练优化
- **数据混合** - 以可配置的权重混合数据集
- **WebDataset 格式** - 从云存储高效流式传输
- **状态管理** - 保存和恢复训练位置

## 安装

```bash
pip install megatron-energon
```

## 主要特性

### 数据处理

- **打包** - 优化序列长度利用率
- **分组** - 对相似长度的序列进行智能批处理
- **连接** - 合并多个数据源
- **对象存储** - 从 S3、GCS、Azure Blob Storage 流式传输

### 生产就绪

- 跨工作进程和节点的分布式加载
- 检查点数据加载状态
- 内存高效的流式传输
- 带预取的并行数据加载

## 基本用法

```python
from megatron.energon import get_train_dataset, get_loader, WorkerConfig

# 创建数据集
ds = get_train_dataset(
    '/path/to/dataset',
    batch_size=32,
    shuffle_buffer_size=1000,
    worker_config=WorkerConfig.default_worker_config(),
)

# 创建加载器并迭代
for batch in get_loader(ds):
    # 训练步骤
    pass
```

## 多模态示例

```python
# 加载图像-文本数据集
ds = get_train_dataset(
    '/path/to/multimodal/dataset',
    batch_size=32,
    worker_config=WorkerConfig(num_workers=8, prefetch_factor=2),
)

for batch in get_loader(ds):
    images = batch['image']  # 图像张量
    texts = batch['text']    # 文本描述
    # 处理批次
```

## 数据集混合

使用自定义权重混合多个数据集：

```python
from megatron.energon import Blender

blended_ds = Blender([
    ('/path/to/dataset1', 0.6),  # 60%
    ('/path/to/dataset2', 0.3),  # 30%
    ('/path/to/dataset3', 0.1),  # 10%
])
```

## 配置

### 工作进程配置

```python
WorkerConfig(
    num_workers=8,              # 并行工作进程数
    prefetch_factor=2,          # 每个工作进程预取的批次数量
    persistent_workers=True,    # 在训练周期之间保持工作进程存活
)
```

### 常用参数

| 参数 | 描述 |
|-----------|-------------|
| `batch_size` | 每批次的样本数 |
| `shuffle_buffer_size` | 随机化缓冲区大小 |
| `max_samples_per_sequence` | 打包到一个序列中的最大样本数 |
| `worker_config` | 并行加载的工作进程配置 |
## 与 Megatron-LM 集成

```python
from megatron.energon import get_train_dataset, get_loader
from megatron.training import get_args

args = get_args()

train_ds = get_train_dataset(
    args.data_path,
    batch_size=args.micro_batch_size,
)

for iteration, batch in enumerate(get_loader(train_ds)):
    loss = train_step(batch)
```

## 资源

- **[Megatron Energon GitHub](https://github.com/NVIDIA/Megatron-Energon)** - 文档和示例
- **[多模态示例](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/multimodal)** - Megatron-LM 多模态训练

## 后续步骤

- 查看[多模态模型](../../models/multimodal.md)了解支持的架构
- 查看[训练示例](../training-examples.md)获取集成示例