<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION 及其许可方保留对本软件、相关文档
   及其任何修改的所有知识产权和专有权利。
   未经 NVIDIA CORPORATION 明示许可协议授权，严禁任何使用、
   复制、披露或分发本软件及相关文档的行为。
-->

# 数据准备

正确准备数据对于使用 Megatron Core 成功训练至关重要。

## 数据格式

Megatron Core 期望训练数据采用 JSONL（JSON Lines）格式，其中每一行都是一个 JSON 对象：

```json
{"text": "Your training text here..."}
{"text": "Another training sample..."}
{"text": "More training data..."}
```

## 预处理数据

使用 `preprocess_data.py` 工具将您的 JSONL 数据转换为 Megatron 的二进制格式：

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

| 参数 | 描述 |
|----------|-------------|
| `--input` | 输入 JSON/JSONL 文件的路径 |
| `--output-prefix` | 输出二进制文件的前缀（.bin 和 .idx 文件） |
| `--tokenizer-type` | 分词器类型（`HuggingFaceTokenizer`、`GPT2BPETokenizer` 等） |
| `--tokenizer-model` | 分词器模型文件的路径 |
| `--workers` | 用于处理的并行工作进程数 |
| `--append-eod` | 添加文档结束（end-of-document）标记 |

## 寻找最佳工作进程数

使用 `--find-optimal-num-workers` 标志来寻找能提供最佳性能（以每秒预处理的文档数衡量）的工作进程数。
脚本将使用不同数量的工作进程启动几次短时间的数据预处理运行，根据收集到的性能数据确定最快的运行。

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --find-optimal-num-workers \
    --workers-to-check 4 8 16 32 \
    --max-documents 50000
```

**必需参数**

| 参数 | 描述 |
|----------|-------------|
| `--find-optimal-num-workers` | 激活最佳工作进程数搜索 |
| `--workers-to-check` | 要测试的可能工作进程数列表 |
| `--max-documents` | 每次运行期间要预处理的文档数量 |

**输出示例**

```bash
-----------------------------------
Performance results (fastest → slowest):
1. 16 workers → avg. docs/s: 9606.6476
2. 32 workers → avg. docs/s: 9275.3284
3. 8 workers → avg. docs/s: 9151.9280
4. 4 workers → avg. docs/s: 6391.3819

-----------------------------------
The most optimal num of workers is 16 with avg. preprocessed docs/s: 9606.6476.
-----------------------------------
```

## 输出文件

预处理工具生成两个文件：
- `processed_data.bin` - 包含分词后序列的二进制文件
- `processed_data.idx` - 用于快速随机访问的索引文件
## 使用预处理数据

在训练脚本中引用您的预处理数据：

```bash
--data-path processed_data \
--split 949,50,1  # Train/validation/test split
```

## 常用分词器

### HuggingFace 分词器

```bash
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model /path/to/tokenizer.model
```

### GPT-2 BPE 分词器

```bash
--tokenizer-type GPT2BPETokenizer \
--vocab-file gpt2-vocab.json \
--merge-file gpt2-merges.txt
```