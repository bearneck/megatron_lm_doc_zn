---
orphan: true
---

<!---
   版权所有 (c) 2022-2026, NVIDIA CORPORATION。保留所有权利。
   NVIDIA CORPORATION 及其许可方保留本软件、相关文档
   及其任何修改的所有知识产权和专有权利。未经 NVIDIA CORPORATION
   明确许可协议授权，严禁任何使用、复制、披露或
   分发本软件及相关文档的行为。
-->

# Megatron 深度探讨

本目录包含关于优化和使用 Megatron 应对各种用例的深度指南、教程和讨论。

## 可用指南

### 训练指南

- **[Megatron-FSDP 用户指南](megatron-fsdp-user-guide/megatron-fsdp-user-guide.md)**

  一份启用 Megatron-FSDP 训练的实用指南，包含 DeepSeek-V3 的快速入门示例、必需和推荐的配置，以及从 torch_dist 到 fsdp_dtensor 的检查点转换说明。

## 贡献指南

如果您想贡献一份指南或教程，请遵循以下结构：

1.  创建一个新目录：`docs/discussions/your-guide-name/`
2.  添加您的主指南：`docs/discussions/your-guide-name/your-guide-name.md`
3.  创建一个图片目录：`docs/discussions/your-guide-name/images/`
4.  更新此 README.md 文件，添加指向您指南的链接

每份指南应自成一体，包含其自身的图片和支持文件。