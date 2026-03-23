<!---
   版权所有 (c) 2022-2026，英伟达公司。保留所有权利。
   英伟达公司及其许可方保留本软件、相关文档及其任何修改的所有知识产权和专有权利。未经英伟达公司明确许可协议授权，严禁任何使用、复制、披露或分发本软件及相关文档的行为。
-->

# 文档开发

- [文档开发](#文档开发)
  - [构建文档](#构建文档)
  - [实时构建](#实时构建)
  - [文档版本](#文档版本)

## 构建文档

以下部分描述了如何设置和构建 NeMo RL 文档。

切换到文档源代码文件夹并生成 HTML 输出。

```sh
cd docs/
uv run --group docs sphinx-build . _build/html
```

* 生成的 HTML 文件位于项目 `docs/` 文件夹下创建的 `_build/html` 文件夹中。
* 生成的 Python API 文档放置在 `docs/` 文件夹下的 `apidocs` 目录中。

## 检查损坏链接

要检查文档中损坏的 http 链接，请运行以下命令：

```sh
cd docs/
uv run --group docs sphinx-build --builder linkcheck . _build/linkcheck
```

它将在 `_build/linkcheck/output.json` 输出一个 JSON 文件，其中包含构建文档时发现的链接。如果链接无法访问，记录的状态将为 `broken`。`docs/conf.py` 文件配置为忽略 GitHub 链接，因为 CI 测试经常会遇到速率限制错误。要检查所有链接，请注释掉那里的 `linkcheck_ignore` 变量。

## 实时构建

在编写文档时，提供一个文档服务并使其在编辑时实时更新会很有帮助。

为此，请运行：

```sh
cd docs/
uv run --group docs sphinx-autobuild . _build/html --port 12345 --host 0.0.0.0
```

打开网络浏览器并访问 `http://${运行_SPHINX_命令的主机}:12345` 以查看输出。

## 文档版本

以下三个文件控制版本切换器。在尝试发布新版本的文档之前，请更新这些文件以匹配最新的版本号。

* docs/versions1.json
* docs/project.json
* docs/conf.py