<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# fusions 包

此包提供包含常用融合操作的模块。融合操作通过增加每次从内存读取张量时完成的工作量来提高计算效率。为了执行融合，此包中的模块要么依赖 PyTorch 的即时编译功能（即旧版 PyTorch 中的 `torch.jit.script` 或新版中的 `torch.compile`），要么调用外部库（如 Apex 或 TransformerEngine）中的自定义内核。