# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""xAI Grok provider with tool support.

Grok uses an OpenAI-compatible API, so this provider reuses the OpenAI
provider's logic with xAI's base URL.
"""

from typing import Optional

from .openai import OpenAIProvider


class GrokProvider(OpenAIProvider):
    """xAI Grok provider with tool calling support.

    Grok's API is OpenAI-compatible, so this extends OpenAIProvider
    with the xAI base URL and default model.

    Supports Grok-1, Grok-2, and other Grok models.
    """

    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-2-latest",
    ):
        """
        Initialize Grok provider.

        Args:
            api_key: xAI API key (or uses XAI_API_KEY env var)
            model: Model to use (e.g., "grok-2-latest", "grok-2", "grok-1")
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Grok provider requires the openai package. "
                "Install with: pip install openai"
            )

        import os

        # Use XAI_API_KEY env var if no key provided
        resolved_key = api_key or os.environ.get("XAI_API_KEY")

        kwargs = {"base_url": self.XAI_BASE_URL}
        if resolved_key:
            kwargs["api_key"] = resolved_key

        self.client = OpenAI(**kwargs)
        self.model = model
