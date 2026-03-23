<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Megatron-LM 中对 Llama、Mistral 及其他类 Llama 模型的支持

注意：为了简化代码，我们现在仅支持转换从 Hugging Face 下载的 llama-3.x 和 mistral 检查点。如需转换其他模型，请参阅 [Megatron Bridge](models/index.md)。

Llama-2 和 Llama-3.x 系列模型是一套开源的预训练和微调（用于聊天）模型，在广泛的基准测试中取得了优异的结果。在发布时，Llama-2 和 Llama-3 模型均取得了开源模型中的最佳成绩，并与领先的闭源模型相媲美（参见 <https://arxiv.org/pdf/2307.09288.pdf>）。

同样地，[Mistral-7b](https://mistral.ai/news/announcing-mistral-7b/) 是一个开源模型，提供预训练和微调（用于聊天）的变体，在基准测试中取得了优异的成绩。

在架构上，Llama-2、Llama-3 和 Mistral-7b 非常相似。因此，Megatron 可以支持加载这三者的检查点进行推理和微调。对于每个模型，转换检查点并加载它们的过程略有不同，下面将分别详细介绍。

# 目录

- [Megatron-LM 中对 Llama、Mistral 及其他类 Llama 模型的支持](#megatron-lm-中对-llamamistral-及其他类-llama-模型的支持)
- [目录](#目录)
- [Llama-2](#llama-2)
  - [下载 Meta 或 Huggingface 检查点](#下载-meta-或-huggingface-检查点)
  - [转换检查点格式](#转换检查点格式)
    - [Meta 格式](#meta-格式)
    - [Huggingface 格式](#huggingface-格式)
  - [启动模型](#启动模型)
    - [启动 Megatron](#启动-megatron)
    - [启动 Meta](#启动-meta)
    - [启动 Huggingface](#启动-huggingface)
  - [基准测试结果](#基准测试结果)
    - [Big Bench](#big-bench)
    - [多语言](#多语言)
    - [LM Evaluation Harness](#lm-evaluation-harness)
    - [MMLU](#mmlu)
- [Llama-3.x](#llama-3x)
  - [下载 Huggingface 检查点](#下载-huggingface-检查点)
  - [转换检查点格式](#转换检查点格式)
    - [Huggingface 格式](#huggingface-格式)
  - [（可选）验证检查点](#可选验证检查点)
  - [启动模型](#启动模型)
- [Mistral-7b](#mistral-7b)
  - [下载 Huggingface 检查点](#下载-huggingface-检查点)
  - [转换检查点格式](#转换检查点格式)
  - [（可选）验证检查点](#可选验证检查点)
  - [启动模型](#启动模型)
- [其他类 Llama 模型支持](#其他类-llama-模型支持)
- [已知的数值差异](#已知的数值差异)
- [使用旧版模型格式](#使用旧版模型格式)
# Llama-2

Llama-2 检查点可以加载到 Megatron 中进行推理和微调。加载这些检查点包括三个步骤：

1.  获取下载检查点的权限。
2.  将检查点从 Meta/Huggingface 格式转换为 Megatron 格式。
3.  设置启动模型的参数。

以下部分详细说明了这些步骤。最后一部分列出了以下两者之间的基准测试结果比较：1) 运行 Meta 格式检查点的 Llama-2 推理代码，和 2) 运行转换后检查点的 Megatron 推理代码。

## 下载 Meta 或 Huggingface 检查点

用户必须首先申请下载 Llama-2 检查点的权限，可以直接从 [Huggingface](https://huggingface.co/docs/transformers/main/model_doc/llama2) (HF) 获取。检查点有两种格式：Meta 原生格式（可从 Meta 和 HF 链接获取）和 HF 格式（仅从 HF 获取）。这两种格式都可以转换为 Megatron 格式，具体细节如下。

## 转换检查点格式

我们建议在训练或微调时传递 `--dtype bf16`。推理可以在 bfloat16 或 float16 下进行。

### Meta 格式

Meta 格式的检查点在转换为 Megatron 格式之前，需要先转换为 HF 格式作为中间步骤。这需要 `transformers` 包，并且版本必须 >=4.31.0（例如，`pip install transformers>=4.31.0`）。（**注意**：我们已专门测试过版本 `4.31.0` 和 `4.32.0`；使用较新版本时您的体验可能有所不同。）假设下载的检查点位于 `$CHECKPOINT_DIR` 中（其中包含 7B、13B、70B 等的单独子目录），可以使用以下示例命令将 Llama-2 格式转换为 bfloat16 的 HF 格式：

```
python tools/checkpoint/convert.py \
>   --model-type GPT \
>   --loader llama_mistral \
>   --load-dir ${META_FORMAT_DIR} \
>   --model-size ${MODEL_SIZE} \
>   --checkpoint-type meta \
>   --tokenizer-model ${TOKENIZER_MODEL} \
>   --saver core \
>   --save-dir ${MEGATRON_FORMAT_DIR} \
>   --target-tensor-parallel-size ${TP} \
>   --target-pipeline-parallel-size ${PP} \
>   --bf16
```

`--model-size` 的有效值为 `llama2-7B`、`llama2-13B` 和 `llama2-70B`（仅适用于预训练模型），以及 `llama2-7Bf`、`llama2-13Bf` 和 `llama2-70Bf`（适用于聊天微调模型）。

### Huggingface 格式

HF 检查点可以通过使用 Megatron 自己的 HF 格式 Llama-2 检查点转换器转换为 Megatron 格式（参见脚本 `tools/checkpoint/loader_llama_mistral.py`）。必须正确设置的一个重要参数是每个模型的张量并行大小 (`TP`)。下表显示了这些值：

| 模型大小 | 张量并行大小 (`TP`) |
| ---------- | --------------------------- |
|  7B        | 1                           |
| 13B        | 2                           |
| 70B        | 8                           |

使用这些 `TP` 值，以及 Llama-2 分词器模型的路径（随原始检查点下载自动下载；参见下面的 `${TOKENIZER_MODEL}`），从您的 Megatron 源代码根目录运行以下命令，将 HF 格式转换为 Megatron 格式：
```
python tools/checkpoint/convert.py \
>   --model-type GPT \
>   --loader llama_mistral \
>   --load-dir ${HF_FORMAT_DIR} \
>   --model-size ${MODEL_SIZE} \
>   --checkpoint-type hf \
>   --tokenizer-model ${TOKENIZER_MODEL} \
>   --saver core \
>   --save-dir ${MEGATRON_FORMAT_DIR} \
>   --target-tensor-parallel-size ${TP} \
>   --target-pipeline-parallel-size ${PP} \
>   --bf16
```

完成此转换后，我们已准备好将检查点加载到 Megatron GPT 模型中。

## 启动模型

### 启动 Megatron

如果用于推理或微调，请使用以下参数：

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 4096 \
--max-position-embeddings 4096 \
--tokenizer-type Llama2Tokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--load ${CHECKPOINT_DIR} \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--use-rotary-position-embeddings \
--normalization RMSNorm \
--no-position-embedding \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32
```

**注意：** 如果您转换到了旧版模型格式（即 `--saver legacy`），请参阅[此处](#using-legacy-model-format)。

### 启动 Meta

Meta 检查点可以通过以下方式启动：<https://github.com/facebookresearch/llama>

### 启动 Huggingface

Huggingface 检查点可以通过以下方式启动：<https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py>

## 基准测试结果

下表列出了原生 Llama-2（使用 Meta 的检查点和 Meta 的推理代码）与 Megatron（使用转换后的 HF 检查点和 Megatron 的推理代码）之间的基准测试比较。

数值是 Megatron 与 Llama-2 之间的百分比误差，使用公式计算：`|<llama_score> - <megatron_score>| / <llama_score>`，其中分数类型在每个表格前详细说明。在所有测试中（每个模型大小共 80 个），平均误差为 0.15%。两个模型之间基准测试分数的微小差异是由于实现中的细微算术差异略微改变了数值。影响这种差异的一些因素包括：

- Megatron 在几个地方执行批量矩阵乘法，例如在自注意力机制和 SwiGLU 中，而 Llama 是分开执行的。
- Megatron 在自注意力机制中使用 `torch.baddbmm`，而 Llama 使用 `torch.matmul`。
- Megatron 使用 `sin`/`cos` 实现旋转位置嵌入，而 Llama 使用 `polar`/`complex` 实现。
- Llama 在初始化期间调用 `torch.set_default_dtype(torch.float16)`，而 Megatron 没有。

### Big Bench

分数类型：多项选择题得分。

| bigbench / 标准 | 7b | 13b | 70b |
| -- | -- | -- | -- |
| date_understanding | 0.29% | 0.13% | 0.12% |
| general_knowledge | 0.00% | 0.00% | 0.00% |
| human_organs_senses | 0.00% | 0.00% | 0.00% |
| intent_recognition | 0.00% | 0.11% | 0.00% |
| riddle_sense | 0.00% | 0.00% | 0.00% |
| similarities_abstraction | 0.00% | 0.58% | 0.00% |
| simple_arithmetic_json_multiple_choice | 0.00% | 0.00% | 0.00% |
| undo_permutation | 0.19% | 0.19% | 0.18% |
### 多语言

评分类型：多项选择题评分。

| 多语言 / xcopa | 7b  | 13b  | 70b |
| -- | -- | -- | -- |
| en-template-mGPT-remove-punctuation | 0.08% | 0.00% | 0.00% |
| et-template-mGPT-remove-punctuation | 0.00% | 0.13% | 0.25% |
| ht-template-mGPT-remove-punctuation | 0.26% | 0.13% | 0.26% |
| id-template-mGPT-remove-punctuation | 0.11% | 0.00% | 0.19% |
| it-template-mGPT-remove-punctuation | 0.00% | 0.10% | 0.09% |
| qu-template-mGPT-remove-punctuation | 0.00% | 0.00% | 0.27% |
| sw-template-mGPT-remove-punctuation | 0.14% | 0.13% | 0.13% |
| th-template-mGPT-remove-punctuation | 0.25% | 0.13% | 0.13% |
| tr-template-mGPT-remove-punctuation | 0.26% | 0.00% | 0.34% |
| vi-template-mGPT-remove-punctuation | 0.00% | 0.11% | 0.00% |
| zh-template-mGPT-remove-punctuation | 0.00% | 0.10% | 0.09% |

### LM 评估工具集

评分类型：多项选择题评分。

| lm-eval | 7b  | 13b  | 70b |
| -- | -- | -- | -- |
| boolq | 0.04% | 0.04% | 0.07% |
| hellaswag | 0.02% | 0.03% | 0.03% |
| piqa | 0.00% | 0.00% | 0.07% |
| winogrande | 0.00% | 0.11% | 0.20% |

### MMLU

评分类型：多项选择题评分。

注意：括号中的数字是每个超类别下的子任务数量。

| mmlu | 7b  | 13b  | 70b |
| -- | -- | -- | -- |
| stem [18]  | 0.79% | 0.05% | 0.01% |
| humanities [13]  | 0.19% | 0.01% | 0.02% |
| other (business, health, misc.) [14]  | 0.08% | 0.06% | 0.12% |
| social sciences [12]  | 0.37% | 0.21% | 0.01% |

# Llama-3.x

Llama-3.x 检查点可以加载到 Megatron 中进行推理和微调。加载这些检查点包含以下几个步骤：

1.  获取下载检查点（权重和分词器）的访问权限。
2.  将检查点从 Huggingface 格式转换为 Megatron 格式。
3.  （可选）验证转换后的检查点
4.  设置启动模型的参数。

以下部分详细说明了这些步骤。

## 下载 Huggingface 检查点

用户必须首先申请从 [Huggingface](https://huggingface.co/meta-llama) 下载 Llama-3.x 检查点的访问权限。

## 转换检查点格式

我们建议在训练或微调时传递 `--dtype bf16`。推理可以在 bfloat16 或 float16 下进行。

### Huggingface 格式

HF 检查点可以使用 Megatron 自带的针对 HF 格式的 Llama-3.x 检查点转换器（参见脚本 `tools/checkpoint/loader_llama_mistral.py`）转换为 Megatron 格式。一个必须正确设置的重要参数是每个模型的张量并行大小 (`TP`)。下表显示了这些值：

| 模型大小 | 张量并行大小 (`TP`) |
| ---------- | --------------------------- |
|  1B        | 1                           |
|  3B        | 1                           |
|  8B        | 1                           |
| 70B        | 8                           |

使用这些 `TP` 值，以及 Llama-3.x 分词器模型的路径（随原始检查点下载自动下载；见下面的 `${TOKENIZER_MODEL}`），从你的 Megatron 源代码根目录运行以下命令，将 HF 格式转换为 Megatron 格式：
```
$>: python tools/checkpoint/convert.py \
 >    --bf16 \
 >    --model-type GPT \
 >    --loader llama_mistral \
 >    --saver core \
 >    --target-tensor-parallel-size ${TP} \
 >    --checkpoint-type hf \
 >    --load-dir ${HF_FORMAT_DIR} \
 >    --save-dir ${MEGATRON_FORMAT_DIR} \
 >    --tokenizer-model ${TOKENIZER_MODEL} \
 >    --model-size llama3 \
```

完成此转换后，我们已准备好将检查点加载到 Megatron GPT 模型中。

## （可选）验证检查点

可以使用脚本 `examples/inference/llama_mistral/run_text_generation_llama3.sh <转换后的核心检查点路径> <下载的Huggingface检查点路径>` 启动一个用于 Llama3 的 Megatron-LM 文本生成服务器。对于 Llama3.1，请使用 `examples/inference/llama_mistral/run_text_generation_llama3.1.sh`。

服务器运行后，使用 `curl 'http://<文本生成服务器IP>:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["<某个提示词>"], "tokens_to_generate":100, "top_k":1}'` 查询服务器。

可以通过运行 `python examples/llama_mistral/huggingface_reference.py --model_path <下载的Huggingface检查点路径> --prompt <某个提示词>` 从 Huggingface transformers 库获取用于比较的参考生成结果。

## 启动模型

如果用于推理或微调，请为 Llama 3.0 使用以下参数：

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 8192 \
--max-position-embeddings 8192 \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--load ${CHECKPOINT_DIR} \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--normalization RMSNorm \
--position-embedding-type rope \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32 \
--disable-bias-linear \
--transformer-impl transformer_engine \
--group-query-attention 8 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--rotary-base 500000 \
--rotary-percent 1.0 \
--ffn-hidden-size 14336 \
--num-attention-heads 32 \
--swiglu \
--bf16 \
```

对于 Llama3.1，请使用以下参数：

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 8192 \
--max-position-embeddings 131072 \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--load ${CHECKPOINT_DIR} \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--normalization RMSNorm \
--position-embedding-type rope \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32 \
--disable-bias-linear \
--transformer-impl transformer_engine \
--group-query-attention 8 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--rotary-base 500000 \
--rotary-percent 1.0 \
--use-rope-scaling \
--ffn-hidden-size 14336 \
--num-attention-heads 32 \
--swiglu \
--bf16 \
```

**注意：** 如果您转换到了旧版模型格式（即 `--saver legacy`），请参阅[此处](#using-legacy-model-format)。
# Mistral-7b

Megatron 目前支持加载 Mistral-7b 的 v0.3 版本（该版本不使用滑动窗口注意力，并提供更大的 32768 词汇表）进行推理和微调。加载这些检查点包含以下几个步骤：

1.  获取下载检查点（权重和分词器）的权限。
2.  将检查点从 HuggingFace 格式转换为 Megatron 格式。
3.  （可选）验证转换后的检查点
4.  设置启动模型的参数。

以下部分详细说明了这些步骤。

## 下载 Huggingface 检查点

用户必须首先申请通过 Huggingface 下载 Mistral-7b 检查点的权限。有两种变体可用：基础模型 ([Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)) 和指令模型 ([Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3))。

## 转换检查点格式

可以使用 Megatron 自带的针对 HF 格式的 Mistral 检查点转换器（参见脚本 `tools/checkpoint/loader_llama_mistral.py`）将 HF 检查点转换为 Megatron 格式。

使用 Mistral 分词器模型的路径（与 HF 检查点一同下载），在 Megatron 源代码根目录下运行以下命令，将格式从 HF 转换为 Megatron 核心格式：

```
$>: python tools/checkpoint/convert.py \
 >    --bf16 \
 >    --model-type GPT \
 >    --loader llama_mistral \
 >    --saver core \
 >    --target-tensor-parallel-size ${TP} \
 >    --checkpoint-type hf \
 >    --load-dir ${HF_FORMAT_DIR} \
 >    --save-dir ${MEGATRON_FORMAT_DIR} \
 >    --tokenizer-model ${TOKENIZER_MODEL} \
 >    --model-size mistral \
```

完成此转换后，我们就可以将检查点加载到 Megatron 核心 GPT 模型中了。

## （可选）验证检查点

可以使用脚本 `examples/inference/llama_mistral/run_text_generation_mistral.sh <PATH_TO_CONVERTED_MCORE_CHECKPOINT> <PATH_TO_DOWNLOADED_HUGGINGFACE_CHECKPOINT>` 启动一个用于 Mistral-7B 的 Megatron-LM 文本生成服务器。

服务器运行后，使用 `curl 'http://<TEXT_GENERATION_SERVER_IP>:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["<SOME_PROMPT>"], "tokens_to_generate":100, "top_k":1}'` 查询服务器。

可以通过运行 `python examples/inference/llama_mistral/huggingface_reference.py --model_path <PATH_TO_DOWNLOADED_HUGGINGFACE_CHECKPOINT> --prompt <SOME_PROMPT>` 从 Huggingface transformers 库获取用于比较的参考生成结果。

## 启动模型

无论是用于推理还是微调，加载时都使用以下参数：

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 4096 \
--max-position-embeddings 4096 \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--load ${CHECKPOINT_DIR} \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--normalization RMSNorm \
--position-embedding-type rope \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32
--apply-layernorm-1p \
--transformer-impl transformer_engine \
--group-query-attention 8 \
--disable-bia-linear \
--rotary-base 1000000 \
--rotary-percent 1.0 \
--swiglu \
--ffn-hidden-size 14336 \
--num-attention-heads 32
```
**注意：** 如果您转换到了旧版模型格式（即使用了 `--saver legacy`），请参阅[此处](#使用旧版模型格式)。

# 其他类 Llama 模型支持

*注意：实验性功能*

许多模型，如 Yi-34B 和 Qwen2.x，都采用了 Llama 架构，可以使用 [Llama-3.x](#llama-3x) 中的命令从 HuggingFace 转换为 Megatron。

# 已知的数值差异

预计 Megatron 和 Huggingface 对 llama3.x 及 mistral 模型的实现不会产生完全相同的数值结果。在多个环节都可能出现微小的数值差异。以下是一个非详尽列表：

1.  TransformerEngine (TE) 在 RMSNorm 内部使用模型参数的数据类型（`params_dtype`），而 Huggingface 的实现使用 fp32。详情请参阅：<https://github.com/NVIDIA/TransformerEngine/issues/1132>
2.  Huggingface `transformers` 将自注意力机制中的 q、k 和 v 投影实现为独立的 GEMM 运算，而 Megatron 核心为了效率将它们合并为一个 GEMM 运算。这导致了微小的数值差异。

# 使用旧版模型格式

本文档中使用的所有检查点转换示例，都使用了保存器格式 `--saver core`，这表示将使用较新（且推荐）的 Megatron GPT 模型类。即：

-   旧类：`megatron.legacy.model.gpt_model.GPTModel`
-   新类：`megatron.core.models.gpt.gpt_model.GPTModel`

使用这种新格式是推荐的方法。但是，如果您的用例需要使用旧类（即，使用 `--saver legacy` 进行转换），那么在启动训练或微调时，必须添加以下参数：

-   `--use-legacy-models`：使用旧版模型类
-   `--ckpt-format torch`：使用 `torch` 检查点格式，这是唯一与旧版模型格式兼容的检查点格式