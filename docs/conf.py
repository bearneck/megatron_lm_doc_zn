# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sphinx documentation builder configuration file.
#
# For a full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Megatron Core"
copyright = "2026, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "nightly"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For our markdown docs
    "sphinx.ext.viewcode",  # Add links to source code in the docs
    "sphinx.ext.doctest",  # Allow testing in docstrings
    "sphinx.ext.napoleon",  # For Google-style docstrings
    "sphinx_copybutton",  # Copy button for code blocks
]

# Check if autodoc generation should be skipped
# Usage: SKIP_AUTODOC=true
skip_autodoc = os.environ.get("SKIP_AUTODOC", "false").lower() == "true"

if not skip_autodoc:
    extensions.append("autodoc2")  # Generate API docs

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for MyST parser (Markdown) --------------------------------------
# MyST parser settings
myst_enable_extensions = [
    "dollarmath",  # Enable $...$ for inline math
    "amsmath",  # Enable display mode LaTeX math
    "colon_fence",  # Enable code blocks using ::: instead of ```
    "deflist",  # Enable definition lists using term: definition
    "fieldlist",  # Enable field lists for metadata (e.g., :author: Name)
    "tasklist",  # Add support for GitHub-style task lists using [ ] and [x]
    "attrs_block",  # Enable setting attributes on block elements using {#id .class key=val}
]
myst_heading_anchors = 5  # Generate anchor links for up to 5 heading levels

# Suppress "more than one target found for cross-reference" warnings for Python symbols that have the same name across multiple modules (e.g., DistributedDataParallelConfig, ModelType).
# These are structural ambiguities in the codebase—cross-references can still resolve; Sphinx just can't automatically pick a unique target.
suppress_warnings = ["ref.python"]

# -- Options for Autodoc2 ---------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

if not skip_autodoc:
    autodoc2_packages = [
        {
            "path": "../megatron/core",  # Path to the package relative to conf.py
            "exclude_dirs": ["converters"],  # List of directory names to exclude
        }
    ]
    autodoc2_render_plugin = "myst"  # Use MyST to render docstrings
    autodoc2_output_dir = "apidocs"  # Output directory for autodoc2 (relative to docs/)
    # This is a workaround to use the parser located in autodoc2_docstrings_parser.py to allow autodoc2 to render Google-style docstrings.
    # Related issue: https://github.com/sphinx-extensions2/sphinx-autodoc2/issues/33
    autodoc2_docstring_parser_regexes = [
        (r".*", "docs.autodoc2_docstrings_parser"),
    ]
    # Regex patterns whose values contain raw regex syntax (e.g., \p{L}) that docutils misparses as footnote/citation markers. Exclude them from generated docs.
    autodoc2_hidden_regexes = [
        r".*\._PATTERN_TIKTOKEN.*",
    ]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "switcher": {
        "json_url": "versions1.json",
        "version_match": release,
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/Megatron-LM/",
            "icon": "fa-brands fa-github",
        }
    ],
    "public_docs_features": True
}
html_extra_path = ["project.json", "versions1.json"]

# Github links are now rate-limited in Github Actions
linkcheck_ignore = [
    ".*github\\.com.*",
    ".*githubusercontent\\.com.*",
]