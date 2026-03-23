<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# 数据流水线

## 数据预处理

数据预处理围绕以下类构建：

1. `IndexedDatasetBuilder`
2. `IndexedDataset`

目前，端到端的数据预处理实现由用户自行完成。更多细节请参阅类的文档字符串。

### IndexedDatasetBuilder

`IndexedDatasetBuilder` 能够构建和合并 `IndexedDataset` 实例。

### IndexedDataset

`IndexedDataset` 类是 Megatron Core 中最低级别的数据接口。在内部，一个 `IndexedDataset` 实例引用两个二进制文件：数据文件（`.bin`）包含文档/序列数据，索引文件（`.idx`）包含文档/序列元数据。

索引文件首先存储数据集级别的元数据：
- 索引头（用于向后兼容）
- 索引版本（用于向后兼容）
- 一个数字代码，对应用于将数据写入数据文件的数据类型
- 数据集中的序列数量
- 数据集中的文档数量

索引文件其次存储文档级别和序列级别的元数据：
- 按顺序，每个序列的元素数量
- 按顺序，每个序列的字节偏移量（指针）
- 按顺序，每个文档的连续序列索引范围 `[...)`
- 按顺序，每个序列的模式（在多模态情况下）

## 数据加载：构建

构建数据加载器是一个分布式感知的过程，围绕以下类构建：

1. `BlendedMegatronDatasetConfig`
2. `BlendedMegatronDatasetBuilder`
3. `IndexedDataset`
3. `MegatronDataset`
4. `BlendedDataset`

更多细节请参阅类的文档字符串。

### BlendedMegatronDatasetConfig（可扩展）

`BlendedMegatronDatasetConfig` 类参数化 `BlendedMegatronDatasetBuilder`，进而参数化 `MegatronDataset` 和 `BlendedDataset`。

不同的训练/推理机制将需要不同的扩展，例如 `GPTDatasetConfig`

### BlendedMegatronDatasetBuilder

`BlendedMegatronDatasetBuilder` 类构建 Megatron Core 中最高级别的数据接口。

**注意：** 所有进程都应尝试通过 `BlendedMegatronDatasetBuilder` 构建数据集，否则程序将挂起。哪些进程最终完成其尝试可以通过 `BlendedMegatronDatasetConfig` 控制。

### IndexedDataset

`IndexedDataset` 类是 Megatron Core 中最低级别的数据接口。

在尝试构建任何高级数据接口之前，`IndexedDataset` 应已存在于磁盘上。

### MegatronDataset（可扩展）

`MegatronDataset` 抽象类是 Megatron Core 中的高级数据接口。它是建立在 `IndexedDataset` 之上的抽象。

不同的训练/推理机制将需要不同的扩展，例如 `GPTDataset`

### BlendedDataset

`BlendedDataset` 类是 Megatron Core 中的高级数据接口。它是建立在 `MegatronDataset` 之上的抽象。

只有当需要混合多个数据分布（即多个 `MegatronDataset` 实例）以贡献到某个数据集分割时，才需要 `BlendedDataset`。混合可以通过 `BlendedMegatronDatasetConfig` 控制。
## 数据加载：实现

### GPTDataset

`GPTDataset` 由以下变量参数化：底层的 `IndexedDataset` 实例 `indexed_dataset`、分割索引 `indexed_indices`（用于训练、验证和测试的文档或序列索引的连续子集）、样本数量 `N`、序列长度 `S` 和随机种子 `R`。

`GPTDataset` 创建了三个索引映射以方便查找：(1) 文档索引，(2) 样本索引，以及 (3) 混洗索引。

1.  文档索引 _Do_idx_ 是一个一维数组，将 _i_ 映射到文档索引，其长度为 `E * |indexed_indices|`，其中 `E` 对应于满足 `E * |indexed_indices| >= N` 的最小周期数。文档索引根据 `R` 进行混洗。

    ```
    给定：

    N = 15
    indexed_indices = [5, 6, 7, 8, 9]
    E = 3

    那么，例如：

    Do_idx = [8, 8, 9, 6, 7, 5, 8, 5, 6, 6, 5, 9, 7, 7, 9]
    ```

2.  样本索引 _Sa_idx_ 是一个二维数组，将 _j_ 映射到 (_i_, _Do_idx_[ _i_ ] 偏移量) 对，其形状为 `[N + 1, 2]`。行 _j_ 和 _j_ + 1 作为第 _j_ 个样本的左右边界。

    ```
    给定：

    S = 1024

    那么，例如：

    Sa_idx[0] = (0, 0)
    Sa_idx[1] = (0, 1024)       => Do_idx[0] 的长度大于 S
    Sa_idx[2] = (1, 512)        => Do_idx[0] 的长度为 1536
    Sa_idx[3] = (2, 0)          => Do_idx[1] 的长度为 1536
    Sa_idx[4] = (5, 300)        => Do_idx[2:5] 相对于 Do_idx[0:2] 是较短的文档
    Sa_idx[5] = (6, 24)         => Do_idx[5] 的长度为 1300
    ```

3.  混洗索引 _Sh_idx_ 是一个一维数组，将 _k_ 映射到 _j_，其长度为 `N`。混洗索引根据 `R` 进行混洗。

    ```
    给定

    N = 10

    那么，例如：

    Sh_idx = [4, 0, 2, 6, 1, 9, 5, 8, 7, 3]
    ```

要查询 `GPTDataset` 的第 _k_ 个样本，我们执行以下操作：

-   使用混洗索引获取样本索引中的索引 _j_。

    ```
    j = Sh_idx[k]
    ```
-   使用样本索引获取文档索引中的左右样本边界索引以及每个文档的起始令牌偏移量。

    ```
    i, offset = Sa_idx[j]
    i_next, offset_next = Sa_idx[j + 1]
    ```
-   使用文档索引从连续的（在文档索引中）文档中检索 `S` 个令牌。

    ```
    sample = []
    sample += indexed_dataset[Do_idx[i]][offset:]
    if i != i_next:
        sample += indexed_dataset[Do_idx[i + 1:i_next]]
    sample += indexed_dataset[Do_idx[i_next]][:offset_next]
    ```

为了在初始化期间节省时间，每个索引在一个进程等级上顺序构建/缓存，随后在其他进程等级上并行加载。缓存的索引对于在 `MegatronDataset.__init__` 函数中生成的哈希值是唯一的。

### BlendedDataset

`BlendedDataset` 由以下变量参数化：底层的 `MegatronDataset` 实例 `D`、权重 `W`（每个数据集一个）和大小 `S`。`BlendedDataset` 将按权重比例从贡献数据集中抽取样本，直到达到所需大小的复合数据集。在每个采样步骤中，我们从具有最大采样误差的数据集中抽取一个样本。
`BlendedDataset` 创建了两个用于辅助查找的"混合"索引：(1) 数据集索引和 (2) 数据集样本索引。

1.  数据集索引 _Da_idx_ 是一个长度为 `S` 的一维数组，将索引 _i_ 映射到数据集索引。

    ```
    给定

    D = [d0, d1, d2]
    W = [1/2, 1/4, 1/4]
    S = 4

    那么，例如：

    Da_idx = [0, 1, 2, 0]

    ```

2.  数据集样本索引 _Sa_idx_ 是一个长度为 `S` 的一维映射，将索引 _i_ 映射到数据集 _Da_idx[i]_ 中的样本索引。

    ```
    给定

    Da_idx = [0, 1, 2, 0]

    那么，例如：

    Sa_idx = [0, 0, 0, 1]
    ```

要查询 `BlendedDataset` 中的第 _k_ 个样本，我们执行以下操作：

-   使用数据集索引从 `D` 中检索对应的数据集，并使用数据集样本索引从该数据集中检索对应的样本。

    ```
    sample = D[Da_idx[k]][Sa_idx[k]]
    ```

为了在初始化时节省时间，每个索引在一个进程上顺序构建/缓存，然后在其他进程上并行加载。缓存的索引对于 `BlendedDataset.__init__` 函数中生成的哈希值是唯一的。

## 快速 DataLoader 初始化

特别是在大规模运行时，DataLoader 的初始化可能需要几分钟，因为它涉及打开和内存映射多个文件，并且可能给文件系统带来显著压力。为了加速这个过程，我们开发了以下三个优化，由配置标志控制：

-   `--dataloader-fast-cache-load`：此选项假设数据集缓存已存在于指定的 `--data-cache-path` 中。启用后，它通过移除同步点和文件检查断言来加速创建过程。

-   `--dataloader-defer-npy-index-mmap`：此选项同样假设数据集缓存已存在于指定的 `--data-cache-path` 中。启用后，它会将数据集索引（.npy 文件）的内存映射推迟到首次访问时进行。我们建议将此配置与 `--num-workers` > 0 一起使用，以便 DataLoader 预取下一批数据，从而隐藏索引内存映射的成本。

-   `--per-dataset-sequences-path`：通过此配置，我们指定由 `tools/build_sequences_per_dataset.py` 脚本生成的 JSON 文件。该脚本生成一个包含所有指定文件前缀所需元数据的单一文件。当处理数百到数千个文件前缀时，此配置特别有用，因为它只需要一次 `open` 操作，而不是每个文件前缀一次。