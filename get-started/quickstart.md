<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 你的首次训练运行

本指南将引导你使用 Megatron Core 运行你的第一个训练任务。请确保在继续之前已完成[安装](install.md)。

## 简单训练示例

在 2 个 GPU 上使用模拟数据运行一个最小的分布式训练循环：

```bash
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

## LLaMA-3 训练示例

在 8 个 GPU 上使用模拟数据，以 FP8 精度训练一个 LLaMA-3 8B 模型：

```bash
./examples/llama/train_llama3_8b_h100_fp8.sh
```

## 数据准备

要在自己的数据上进行训练，Megatron 期望使用预处理后的二进制文件（`.bin` 和 `.idx`）。

### 1. 准备一个 JSONL 文件

每一行应包含一个 `text` 字段：

```json
{"text": "你的训练文本在这里..."}
{"text": "另一个训练样本..."}
```

### 2. 预处理数据

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

### 关键参数

- `--input`: 输入 JSON/JSONL 文件的路径
- `--output-prefix`: 输出二进制文件的前缀（.bin 和 .idx）
- `--tokenizer-type`: 分词器类型（`HuggingFaceTokenizer`、`GPT2BPETokenizer` 等）
- `--tokenizer-model`: 分词器模型文件的路径
- `--workers`: 用于处理的并行工作进程数
- `--append-eod`: 添加文档结束标记

## 后续步骤

- 探索[并行策略](../user-guide/parallelism-guide.md)以扩展你的训练规模
- 了解[数据准备](../user-guide/data-preparation.md)的最佳实践
- 查看[高级功能](../user-guide/features/index.md)以了解高级能力