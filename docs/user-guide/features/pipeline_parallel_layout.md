<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 自定义流水线模型并行布局

*这是一个实验性功能，可能会发生变化。*

`--pipeline-model-parallel-layout` 是一个用于定义流水线并行分区的灵活 API，这对于为不平衡模型实现平衡分区至关重要。例如，要使用 PP16VPP2 对 DeepSeek-V3（61 个解码器层 + 1 个 mtp 层）进行分区，我们可以包含以下参数：

```bash
--pipeline-model-parallel-size 16
--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"
```

| PP \ VPP 排名 |            0            |       1       |
|---------------|-------------------------|---------------|
|       0       | 嵌入层 + 3 × 解码器层 |  2 × 解码器层  |
|      1~13     |        2 × 解码器层      |  2 × 解码器层  |
|       14      |        2 × 解码器层      |      mtp 层      |
|       15      |        2 × 解码器层      |      损失计算层     |

在布局字符串中，阶段由 '|' 分隔。可以复制的阶段或层可以用乘法表示。逗号可以用于美观。符号选择：

* `E` = 嵌入层
* `t` = 变换器解码器层
* `m` = MTP 层
* `L` = 损失计算层

请注意，允许存在空阶段，例如 `E||t|L`（第二个阶段为空）。