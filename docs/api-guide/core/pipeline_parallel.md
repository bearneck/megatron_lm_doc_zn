<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION 及其许可方保留本软件、相关文档
   及其任何修改的全部知识产权和专有权利。未经 NVIDIA CORPORATION 明确许可，
   严禁任何使用、复制、披露或分发本软件及相关文档的行为。
-->

# pipeline_parallel 包

本包包含两种不同流水线并行调度方案的实现（一种无交错，一种带交错，详见[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)），以及一个默认的非流水线调度方案。同时，它还包含了流水线阶段间所需的点对点通信方法。