<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 优化器 CPU 卸载

## 如何使用？

在 MCore 中添加以下标志以启用优化器 CPU 卸载。

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--use-precision-aware-optimizer
```

## 配置建议

梯度从 GPU 复制到 CPU、CPU 优化器步骤以及随后参数从 CPU 复制到 GPU 可能是耗时的操作，建议使用标志 `--overlap-cpu-optimizer-d2h-h2d` 来并发执行它们。
