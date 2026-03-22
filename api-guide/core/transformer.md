<!---
   版权所有 (c) 2022-2026, NVIDIA CORPORATION。保留所有权利。
   NVIDIA CORPORATION 及其许可方保留本软件、相关文档
   及其任何修改的所有知识产权和专有权利。
   未经 NVIDIA CORPORATION 明确许可协议授权，严禁任何使用、
   复制、披露或分发本软件及相关文档的行为。
-->

# transformer 包

`transformer` 包提供了一个可定制和可配置的
transformer 模型架构实现。transformer 堆栈的每个组件，
从整个层到单个线性层，都可以通过使用 "spec" 参数
交换不同的 PyTorch 模块来进行定制。transformer 的
配置（隐藏层大小、层数、注意力头数等）通过
`TransformerConfig` 对象提供。