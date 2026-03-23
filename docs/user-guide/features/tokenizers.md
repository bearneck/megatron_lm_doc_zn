<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Tokenizers

Megatron Core 提供了一个统一的 tokenizer 系统，具有类似 HuggingFace 的 API，便于 tokenizer 的管理和配置。

## 概述

`MegatronTokenizer` 类提供了一个简单、熟悉的 API 用于加载和管理 tokenizer：

- **自动检测** - 无需指定库即可加载任何类型的 tokenizer
- **基于元数据的配置** - 将 tokenizer 设置存储在 JSON 中以便重用
- **兼容 HuggingFace 的 API** - 熟悉的 `.from_pretrained()` 接口
- **自定义 tokenizer 支持** - 可扩展模型特定的分词逻辑

## 主要特性

### 统一 API

无论 tokenizer 后端如何（SentencePiece、HuggingFace、TikToken 等），都使用相同的 API：

```python
from megatron.core.tokenizers import MegatronTokenizer

tokenizer = MegatronTokenizer.from_pretrained("/path/to/tokenizer")
```

### Tokenizer 元数据

配置存储在一个 JSON 元数据文件中，包含：
- Tokenizer 库（HuggingFace、SentencePiece、TikToken 等）
- 聊天模板
- 自定义 tokenizer 类
- 特殊 token 配置

**优点：**
- 配置一次，随处重用
- 无需重复的 CLI 参数
- 易于共享 - 只需复制 tokenizer 目录

### 自动库检测

自动选择正确的 tokenizer 实现：
- 无需指定 `SentencePieceTokenizer`、`HuggingFaceTokenizer` 等
- 从元数据中检测库类型
- 在 tokenizer 后端之间无缝切换

## 基本用法

### 创建 Tokenizer 元数据

保存 tokenizer 配置以便重用：

```python
from megatron.core.tokenizers import MegatronTokenizer

# Create metadata for a SentencePiece tokenizer
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/tokenizer.model",
    tokenizer_library="sentencepiece",
    chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
)
```

元数据将作为 `tokenizer_metadata.json` 保存在 tokenizer 目录中。

### 加载 Tokenizer

从包含元数据的目录加载：

```python
from megatron.core.tokenizers import MegatronTokenizer

# Load with auto-detected configuration
tokenizer = MegatronTokenizer.from_pretrained("/path/to/tokenizer.model")
```

### 使用自定义元数据路径加载

如果元数据单独存储：

```python
tokenizer = MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/tokenizer.model",
    metadata_path="/path/to/custom/metadata.json",
)
```

### 使用内联元数据加载

将元数据作为字典传递：

```python
tokenizer = MegatronTokenizer.from_pretrained(
    tokenizer_path="GPT2BPETokenizer",
    metadata_path={"library": "megatron"},
    vocab_file="/path/to/vocab.txt",
)
```
## 高级用法

### 自定义分词器类

创建模型特定的分词逻辑：

```python
from megatron.core.tokenizers.text import MegatronTokenizerText

class CustomTokenizer(MegatronTokenizerText):
    def encode(self, text):
        # Custom encoding logic
        return super().encode(text)

    def decode(self, tokens):
        # Custom decoding logic
        return super().decode(tokens)

# Save metadata with custom class
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/tokenizer.model",
    tokenizer_library="sentencepiece",
    tokenizer_class=CustomTokenizer,
)
```

### TikToken 分词器

配置基于 TikToken 的分词器：

```python
tokenizer = MegatronTokenizer.from_pretrained(
    tokenizer_path="/path/to/tokenizer/model.json",
    metadata_path={"library": "tiktoken"},
    pattern="v2",
    num_special_tokens=1000,
)
```

### 空分词器

为测试或非文本模型使用空分词器：

```python
tokenizer = MegatronTokenizer.from_pretrained(
    metadata_path={"library": "null-text"},
    vocab_size=131072,
)
```

## 与 Megatron-LM 集成

### 在训练脚本中使用

分词器系统与 Megatron-LM 训练无缝集成：

```bash
# Null tokenizer for testing
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072 \
    ...
```

```bash
# HuggingFace tokenizer with metadata
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Meta-Llama-3-8B \
    --tokenizer-metadata /path/to/metadata.json \
    ...
```

### 自动生成的元数据

如果未指定 `--tokenizer-metadata`，系统会根据分词器类型自动生成一个默认的元数据文件。

## 支持的分词器库

| 库 | 描述 | 使用场景 |
|---------|-------------|----------|
| **HuggingFace** | Transformers 分词器 | 大多数现代 LLM（LLaMA、Mistral 等） |
| **SentencePiece** | 谷歌的分词器 | GPT 风格模型，自定义词汇表 |
| **TikToken** | OpenAI 的分词器 | GPT-3.5/GPT-4 风格的分词 |
| **Megatron** | 内置分词器 | 传统的 GPT-2 BPE |
| **Null** | 无操作分词器 | 测试、非文本模态 |

## 常见分词器类型

### LLaMA / Mistral

```python
MegatronTokenizer.write_metadata(
    tokenizer_path="/path/to/llama/tokenizer.model",
    tokenizer_library="sentencepiece",
)
```

### GPT-2

```python
MegatronTokenizer.write_metadata(
    tokenizer_path="GPT2BPETokenizer",
    tokenizer_library="megatron",
    vocab_file="/path/to/gpt2-vocab.json",
    merge_file="/path/to/gpt2-merges.txt",
)
```

## 最佳实践

1. **始终保存元数据** - 创建一次元数据，在多次训练运行中重复使用
2. **使用 HuggingFace 分词器** - 在可能的情况下，以获得现代 LLM 的兼容性
3. **测试分词** - 在开始训练前验证编码/解码功能
4. **版本控制元数据** - 将 `tokenizer_metadata.json` 包含在你的实验配置中
5. **共享分词器目录** - 为了可复现性，同时包含模型文件和元数据
## 后续步骤

- **准备数据**：查看[数据准备](../data-preparation.md)了解如何使用分词器进行预处理
- **训练模型**：在[训练示例](../training-examples.md)中使用分词器
- **支持的模型**：查看[语言模型](../../models/llms.md)获取特定模型的分词器信息