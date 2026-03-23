---
orphan: true
---

<!---
   版权所有 (c) 2022-2026, NVIDIA CORPORATION。保留所有权利。
   NVIDIA CORPORATION 及其许可方保留对本软件、相关文档
   及其任何修改的所有知识产权和专有权利。未经 NVIDIA CORPORATION
   明确许可协议授权，严禁任何使用、复制、披露或分发本软件及相关文档的行为。
-->

# API 向后兼容性检查

## 概述

Megatron Core 使用自动化的 API 兼容性检查来确保不同版本之间的接口稳定。这可以防止意外引入破坏性变更，从而影响用户在版本间升级。

## 工作原理

兼容性检查器会：
1. 将当前代码与最新发布版本进行比较
2. 检测函数签名中的破坏性变更
3. 如果发现破坏性变更，则使 CI 失败（除非明确豁免）
4. 在每次修改 `megatron/core` 的 PR 上自动运行

## 检查内容

### ✅ 检测到的破坏性变更

- **参数被移除** - 移除函数参数
- **添加了无默认值的参数** - 添加必需参数
- **参数顺序改变** - 改变参数的顺序
- **可选→必需** - 移除参数的默认值
- **函数被移除** - 删除公共函数
- **返回类型改变** - 更改返回类型注解（警告）

### ⏭️ 被跳过的内容

- **测试函数** - 以 `test_` 开头的函数
- **豁免装饰器** - 标记有 `@internal_api`、`@experimental_api` 或 `@deprecated` 的函数
- **排除路径** - `tests/`、`experimental/`、`legacy/` 目录下的代码

### ✅ 允许的变更

- **添加可选参数** - 添加带有默认值的参数
- **添加新函数** - 新的公共 API
- **使参数变为可选** - 为必需参数添加默认值

## 给开发者的指南

### 本地运行

```bash
# Install griffe
pip install griffe

# Check against latest release
python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0

# Check with verbose output
python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 -v

# Compare two specific branches
python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 --current main
```

### 将函数标记为豁免

如果您需要对内部或实验性 API 进行破坏性变更：

#### 内部 API（用于内部实现细节）

```python
from megatron.core.utils import internal_api

@internal_api
def experimental_feature(x, y):
    """
    This API is experimental and may change.
    NOT FOR EXTERNAL USE.
    """
    pass
```

**何时使用 `@internal_api`：**
- 未对外部使用文档化的内部 API
- 明确标记为不稳定的实验性功能
- 尚未发布、正在开发中的函数

#### 实验性 API（用于实验性功能）

```python
from megatron.core.utils import experimental_api

@experimental_api
def new_experimental_feature(x, y):
    """
    This API is experimental and may change without notice.
    """
    pass
```
**何时使用 `@experimental_api`：**
- 明确标记为不稳定的实验性功能
- 正在积极开发的新 API
- 尚未稳定的功能

### 弃用 API

对于计划中的 API 变更，请使用弃用工作流：

```python
from megatron.core.backwards_compatibility_decorators import deprecated

@deprecated(
    version="1.0.0",           # When deprecation starts
    removal_version="2.0.0",    # When it will be removed
    alternative="new_function", # Recommended replacement
    reason="Improved performance and cleaner API"
)
def old_function(x):
    """This function is deprecated."""
    pass
```

**弃用时间线：**
1.  **版本 N** - 添加 `@deprecated` 装饰器，函数仍可工作
2.  **版本 N+1** - 保留函数并发出弃用警告
3.  **版本 N+2** - 移除函数（用户已收到警告）

### 处理 CI 失败

如果兼容性检查在你的 PR 上失败：

1.  **在 CI 日志中审查破坏性变更**
2.  **选择一个操作：**
    -   **修复代码** - 回退破坏性变更
    -   **添加豁免** - 如果是故意的，使用 `@internal_api`
    -   **使用弃用** - 对于计划中的 API 变更
3.  **用修复方案更新你的 PR**

## 示例

### 示例 1：兼容性变更

```python
# ✅ BEFORE (v1.0)
def train_model(config, dataloader):
    pass

# ✅ AFTER (v1.1) - Added optional parameter
def train_model(config, dataloader, optimizer="adam"):
    pass
```
**结果：** ✅ 检查通过

---

### 示例 2：破坏性变更

```python
# BEFORE (v1.0)
def train_model(config, dataloader, optimizer="adam"):
    pass

# ❌ AFTER (v1.1) - Removed parameter
def train_model(config, dataloader):
    pass
```
**结果：** ❌ 检查失败 - "参数 'optimizer' 被移除"

---

### 示例 3：豁免内部 API

```python
from megatron.core.utils import internal_api

# BEFORE (v1.0)
@internal_api
def _internal_compute(x, y):
    pass

# ✅ AFTER (v1.1) - Can change freely
@internal_api
def _internal_compute(x, y, z):  # Added parameter
    pass
```
**结果：** ✅ 检查通过（函数被豁免）

---

### 示例 4：弃用工作流

```python
from megatron.core.utils import deprecated

# Version 1.0 - Add deprecation
@deprecated(
    version="1.0.0",
    removal_version="2.0.0",
    alternative="train_model_v2"
)
def train_model(config):
    """Old training function - DEPRECATED"""
    pass

def train_model_v2(config, **options):
    """New improved training function"""
    pass

# Version 1.1 - Keep both (users migrate)
# Version 2.0 - Remove train_model()
```

## 架构

```
Developer commits code
    ↓
GitHub Actions triggers
    ↓
CI runs check_api_backwards_compatibility.py
    ↓
Script loads code via griffe:
  • Baseline: latest release (e.g., core_r0.8.0)
  • Current: PR branch
    ↓
Apply filtering:
  • Skip @internal_api, @experimental_api, and @deprecated
  • Skip private functions (_prefix)
  • Skip test/experimental paths
    ↓
Griffe compares signatures:
  • Parameters
  • Types
  • Return types
  • Defaults
    ↓
Report breaking changes
    ↓
Exit: 0=pass, 1=fail
    ↓
CI fails if breaking changes detected
```
## 配置

### 自定义过滤器

编辑 `scripts/check_api_backwards_compatibility.py`：

```python
# Add more exempt decorators
EXEMPT_DECORATORS = [
    "internal_api",
    "experimental_api",
    "deprecated",
]

# Add more path exclusions
EXCLUDE_PATHS = {
    "tests",
    "experimental",
    "legacy",
    "your_custom_path",  # ← Add here
}
```

### 更改基线

工作流会自动检测最新的 `core_r*` 标签。如需手动指定：

```yaml
# In .github/workflows/check_api_backwards_compatibility_workflow.yml
- name: Run compatibility check
  run: |
    python scripts/check_api_backwards_compatibility.py \
      --baseline your_custom_baseline
```

## 常见问题

### 问：为什么我的 PR 未能通过兼容性检查？

**答：** 你的代码与上一个版本相比引入了破坏性变更。请查看 CI 日志以了解具体更改。

### 问：我可以为我的 PR 禁用此检查吗？

**答：** 不可以，但你可以使用 `@internal_api` 或 `@experimental_api` 将特定函数标记为豁免。

### 问：如果我需要进行破坏性变更怎么办？

**答：** 使用 `@deprecated` 装饰器进行渐进式过渡，或者使用 `@internal_api`（针对内部代码）或 `@experimental_api`（针对实验性功能）将函数标记为豁免。

### 问：这会检查整个 Megatron-LM 吗？

**答：** 不会，仅检查 `megatron/core/**`（Megatron Core）。遗留代码已被排除。

### 问：类方法呢？

**答：** 是的，类方法会像函数一样被检查。

### 问：我可以在推送代码前在本地运行此检查吗？

**答：** 可以！运行 `python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0`

### 问：如果还没有发布标签怎么办？

**答：** 工作流将使用 `main` 作为基线。一旦有了发布标签，请更新它。

## 故障排除

### 错误："griffe is not installed"

```bash
pip install griffe
```

### 错误："No core_r* tags found"

仓库还没有发布标签。工作流将回退到使用 `main`。

### 误报

如果检查器报告了一个实际上并非破坏性的变更，请提交一个问题，并使用 `@internal_api` 作为临时解决方案。

## 参考

- **脚本：** `scripts/check_api_backwards_compatibility.py`
- **工作流：** `.github/workflows/check_api_backwards_compatibility_workflow.yml`
- **装饰器：** `megatron/core/backwards_compatibility_decorators.py`
- **Griffe 文档：** https://mkdocstrings.github.io/griffe/

## 支持

如有问题或疑问：
1. 查阅本文档
2. 查看已有的包含兼容性检查的 PR
3. 在 Megatron-LM Slack/Discord 中提问
4. 在 GitHub 上提交问题