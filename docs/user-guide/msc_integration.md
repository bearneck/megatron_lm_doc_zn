<!---
   版权所有 (c) 2022-2026, NVIDIA CORPORATION。保留所有权利。
   NVIDIA CORPORATION 及其许可方保留本软件、相关文档
   及其任何修改的所有知识产权和专有权利。
   未经 NVIDIA CORPORATION 明确许可协议，严禁任何使用、复制、披露或
   分发本软件及相关文档的行为。
-->

# 多存储客户端（MSC）集成

[多存储客户端](https://github.com/NVIDIA/multi-storage-client)（MSC）提供了一个统一的接口，用于从文件系统（例如本地磁盘、NFS、Lustre）和对象存储提供商（如 S3、GCS、OCI、Azure、AIStore 和 SwiftStack）读取数据集和存储检查点。

本指南将引导您了解如何：

1. 如何安装和配置 MSC
2. 如何直接使用对象存储中的数据集训练模型
3. 如何将模型检查点保存到对象存储以及从对象存储加载

## 安装

MSC 在 PyPI 上以 `multi-storage-client` 包的形式提供。

基础 [客户端](https://nvidia.github.io/multi-storage-client/user_guide/concepts.html#term-client) 默认支持 POSIX 文件系统，但每个存储服务都有额外的扩展包，为其对应的存储提供商提供必要的包依赖。

```bash
# POSIX 文件系统。
pip install multi-storage-client

# AWS S3 和 S3 兼容的对象存储。
pip install "multi-storage-client[boto3]"

# Google Cloud Storage (GCS)。
pip install "multi-storage-client[google-cloud-storage]"
```

## 配置文件

MSC 使用一个 YAML 配置文件来定义如何连接到对象存储系统。这种设计允许您指定一个或多个存储配置文件，每个配置文件代表一个不同的存储后端或存储桶。MSC 通过将详细信息集中在配置文件中，使您的训练脚本保持简洁和可移植。无需将访问密钥、存储桶名称或其他提供商特定选项硬编码到您的代码中。

以下是一个配置示例：

```yaml
profiles:
  my-profile:
    storage_provider:
      type: s3
      options:
        # 将存储桶/容器名称设置为 base_path
        base_path: my-bucket
        region_name: us-west-2
    # 可选凭据（对于 S3 也可以使用环境变量）
    credentials_provider:
      type: S3Credentials
      options:
        access_key: ${AWS_ACCESS_KEY}
        secret_key: ${AWS_SECRET_KEY}

cache:
  size: 500G               # 最大缓存大小
  location: /tmp/msc_cache # 文件系统上的缓存目录
```

要告诉 MSC 在哪里找到此文件，请在运行 Megatron-LM 脚本之前设置以下环境变量：

```bash
export MSC_CONFIG=/path/to/msc_config.yaml
```

## MSC URL 格式

MSC 使用自定义 URL 方案来识别和访问不同对象存储提供商中的文件。该方案使得引用数据和检查点变得容易，而无需担心底层的存储实现。MSC URL 具有以下结构：

```
msc://<profile-name>/<path/to/object>
```

**组成部分：**

* `msc://` 这是方案标识符，表示该路径应由多存储客户端解释。
* `<profile-name>` 这对应于您在 YAML 配置文件的 profiles 部分下定义的命名配置文件。每个配置文件指定存储提供商（例如 S3、GCS）、凭据以及存储特定选项，例如存储桶名称或基础路径。
* `<path/to/object>` 这是存储提供商内对象或目录的逻辑路径，相对于配置文件中配置的 base_path。它的行为类似于本地文件系统中的路径，但映射到底层存储系统中的对象键或 blob。
**示例：**

给定以下配置文件：

```yaml
profiles:
  my-profile:
    storage_provider:
      type: s3
      options:
        base_path: my-bucket
```

MSC URL：

```
msc://my-profile/dataset/train/data.bin
```

被解释为访问名为 `my-bucket` 的 S3 存储桶内键为 `dataset/train/data.bin` 的对象。如果这是一个 GCS 或 OCI 配置文件，MSC 将根据配置文件定义应用相应的后端逻辑，但您使用 MSC URL 的代码将保持不变。

这种抽象允许训练脚本统一引用存储资源——无论它们托管在 AWS、GCP、Oracle 还是 Azure 上——只需在配置文件中切换配置文件即可。


## 从对象存储训练

要使用存储在对象存储中的数据集进行训练，请将 MSC URL 与 `--data-path` 参数一起使用。此 URL 引用在您的 MSC 配置文件中定义的配置文件下存储的数据集。

此外，Megatron-LM 在从对象存储读取时需要 `--object-storage-cache-path` 参数。此路径用于缓存与 IndexedDataset 关联的 `.idx` 索引文件，这些文件对于高效数据访问是必需的。

```bash
python pretrain_gpt.py                                      \
    --object-storage-cache-path /path/to/object_store_cache \
    --data-cache-path /path/to/data_cache                   \
    --data-path msc://my-profile/datasets/text_document     \
    --no-mmap-bin-files
```

**注意：** 使用 MSC 从对象存储中的数据集进行训练时，必须提供所有四个参数。

## 从对象存储保存和加载检查点

通过为 `--save` 和 `--load` 参数指定 MSC URL，可以使用 MSC 直接从对象存储保存和加载模型检查点。这允许您在对象存储中管理检查点。

```bash
python pretrain_gpt.py                \
  --save msc://my-profile/checkpoints \
  --load msc://my-profile/checkpoints \
  --save-interval 1000
```

**注意：** 保存到或从 MSC URL 加载时，目前仅支持 `torch_dist` 检查点格式。

## 禁用 MSC

默认情况下，当安装了 `multi-storage-client` 库时，MSC 集成会自动启用。即使不使用显式的 MSC URL，MSC 也用于常规文件系统路径（例如 `--data-path`、`--save` 或 `--load` 中的 `/filesystem_mountpoint/path`）。MSC 作为一个非常薄的抽象层，当与常规路径一起使用时，其性能影响可以忽略不计，因此通常无需禁用它。如果您需要禁用 MSC，可以使用 `--disable-msc` 标志：

```bash
python pretrain_gpt.py --disable-msc
```

## 性能考虑

当使用带有 MSC 的对象存储时，需要记住一些重要的性能影响：

**读取数据集**

直接从对象存储读取训练数据集通常比从本地磁盘读取慢。这主要是由于：
* 对象存储系统的高延迟，特别是对于小型和随机读取操作（例如，从 .bin 文件中读取样本）。
* 对象存储使用的基于 HTTP 的协议（例如，带有范围请求的 S3 GET），比本地文件系统 I/O 慢。
为了补偿这种延迟，建议在训练命令中使用 `--num-workers` 参数增加数据加载工作进程的数量：

```
python pretrain_gpt.py --num-workers 8 ...
```

增加工作进程数量允许从对象存储进行更多并行读取，有助于掩盖 I/O 延迟，并在训练期间保持较高的 GPU 利用率。

**检查点加载**

当使用 MSC 从对象存储加载检查点时，在 MSC 配置文件中正确配置缓存部分非常重要。此本地缓存用于存储下载的检查点数据和元数据，这可以显著减少加载时间和内存使用。

示例：

```
cache:
  size: 500G
  location: /tmp/msc_cache
```

为了获得最佳性能，请将缓存目录配置在高速本地存储设备上，例如 NVMe SSD。

## 其他资源和高级配置

有关 MSC 配置选项的完整文档，包括支持的存储提供商、凭据管理和高级缓存策略的详细信息，请参阅 [MSC 配置文档](https://nvidia.github.io/multi-storage-client/references/configuration.html)。

MSC 支持收集可观测性指标和跟踪，以帮助监控和调试训练期间的数据访问模式。这些指标可以帮助您识别数据加载流水线中的瓶颈、优化缓存策略，并在使用对象存储训练大型数据集时监控资源利用率。有关 MSC 可观测性功能的更多信息，请参阅 [MSC 可观测性文档](https://nvidia.github.io/multi-storage-client/user_guide/telemetry.html)。

MSC 提供了一个实验性的 Rust 客户端，它绕过了 Python 的全局解释器锁（GIL），从而显著提高了多线程 I/O 操作的性能。Rust 客户端支持 AWS S3、SwiftStack 和 Google Cloud Storage，能够实现真正的并发执行，与 Python 实现相比性能提升显著。要启用它，请在存储提供商配置中添加 `rust_client: {}`。更多详细信息，请参阅 [MSC Rust 客户端文档](https://nvidia.github.io/multi-storage-client/user_guide/rust.html)。