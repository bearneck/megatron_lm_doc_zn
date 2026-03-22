<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 如何提交 PR

所有 PR 最初都处于 **草稿** 状态。如果您打开一个非草稿状态的 PR，它将被自动转换为草稿。

## 步骤 1：将 PR 标记为"准备就绪，等待审查"

1.  当您的 PR 准备就绪时，点击 **准备就绪，等待审查**。
2.  值班审查员会自动分配，并且专家审查员会根据您的更改收到通知。他们将收到通知并很快处理您的 PR。

:warning: 请确保在标记为就绪之前，所有合并冲突都已解决，并且 CI 测试已通过。
如果未满足这些要求，最终审查可能会被拒绝。

## 步骤 2：最终审查（仅限 `megatron/core`）

对于修改 `megatron/core` 的 PR，一旦所有专家审查员都已批准，`Final Review` 标签将**自动**应用，并分配最终审查员。

对于 `megatron/core` 之外的 PR，此步骤将被跳过。

## 步骤 3：已批准

一旦所有必需的审查员都已批准，`Approved` 标签将**自动**应用。此时 PR 已准备好合并。

## 步骤 4：合并

[mcore-engineers](https://github.com/orgs/NVIDIA/teams/mcore-engineers) 的任何成员都将能够合并您的 PR。