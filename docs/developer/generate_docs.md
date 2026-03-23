<!---
   版权所有 (c) 2022-2026, NVIDIA CORPORATION。保留所有权利。
   NVIDIA CORPORATION 及其许可方保留本软件、相关文档
   及其任何修改的所有知识产权和专有权利。
   未经 NVIDIA CORPORATION 明确许可协议授权，严禁任何使用、
   复制、披露或分发本软件及相关文档的行为。
-->

# 本地生成文档

要在本地生成文档，请使用以下命令：

```
cd docs
uv run --only-group docs sphinx-autobuild . _build/html --port 8080 --host 127.0.0.1
```

文档将在 <http://localhost:8080/> 生成。

**推荐：** 在生成文档时设置环境变量 `SKIP_AUTODOC=true` 以跳过 `apidocs` 的生成。