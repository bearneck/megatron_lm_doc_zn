# 版权所有 (c) 2025, NVIDIA CORPORATION. 保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非遵守许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按"原样"分发，不附带任何明示或暗示的担保或条件。
# 请参阅许可证中规定的特定语言管理权限和限制。

from docutils import nodes
from myst_parser.parsers.sphinx_ import MystParser
from sphinx.ext.napoleon.docstring import GoogleDocstring


class NapoleonParser(MystParser):
    """添加对 Google 风格文档字符串的支持。"""

    def parse(self, input_string: str, document: nodes.document) -> None:
        """解析 Google 风格的文档字符串。"""

        # 获取 Sphinx 配置
        config = document.settings.env.config

        # 使用 Google 风格进行处理
        google_parsed = str(GoogleDocstring(input_string, config))

        return super().parse(google_parsed, document)


Parser = NapoleonParser