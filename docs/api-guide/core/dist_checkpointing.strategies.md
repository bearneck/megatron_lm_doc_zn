<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# dist_checkpointing.strategies 包

该包定义了不同的检查点格式（后端）以及保存/加载算法（策略）。

策略可用于实现新的检查点格式，或为现有格式实现新的（针对特定用例更优的）保存/加载方式。
策略被传递给 `dist_checkpointing.load` 和 `dist_checkpointing.save` 函数，并控制实际的保存/加载过程。