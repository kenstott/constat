# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Prompt and template loading utilities."""

from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(filename: str) -> str:
    """Load a prompt/template from the prompts directory."""
    path = _PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_yaml(filename: str) -> dict:
    """Load a YAML file from the prompts directory."""
    path = _PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
