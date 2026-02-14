# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Google Gemini provider with tool support."""

from typing import Callable, Optional

from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider with tool calling support.

    Supports Gemini Pro, Gemini Ultra, and other Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-pro",
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key (or uses GOOGLE_API_KEY env var)
            model: Model to use (e.g., "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro")
        """
        try:
            # noinspection PyUnresolvedReferences
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Gemini provider requires the google-generativeai package. "
                "Install with: pip install google-generativeai"
            )

        if api_key:
            genai.configure(api_key=api_key)

        self.genai = genai
        self.model_name = model

    @staticmethod
    def _convert_tools_to_gemini_format(tools: list[dict]) -> list:
        """Convert Anthropic-style tools to Gemini function declarations.

        Anthropic format:
            {"name": "x", "description": "y", "input_schema": {...}}

        Gemini format uses FunctionDeclaration objects.
        """
        # noinspection PyUnresolvedReferences
        from google.generativeai.types import FunctionDeclaration

        function_declarations = []
        for tool in tools:
            # Extract parameters from input_schema
            schema = tool.get("input_schema", {"type": "object", "properties": {}})

            function_declarations.append(
                FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=schema,
                )
            )
        return function_declarations

    def generate(
        self,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate a response, automatically handling tool calls.

        Converts Anthropic-style tool definitions to Gemini format.
        """
        # noinspection PyUnresolvedReferences
        from google.generativeai.types import Tool

        tool_handlers = tool_handlers or {}
        use_model = model or self.model_name

        # Create model with system instruction
        generation_config = {
            "max_output_tokens": max_tokens,
        }

        model_instance = self.genai.GenerativeModel(
            model_name=use_model,
            system_instruction=system,
            generation_config=generation_config,
        )

        # Convert and set up tools
        gemini_tools = None
        if tools:
            function_declarations = self._convert_tools_to_gemini_format(tools)
            gemini_tools = [Tool(function_declarations=function_declarations)]

        # Start chat session
        chat = model_instance.start_chat(history=[])

        # Send initial message
        # noinspection PyTypeChecker
        kwargs = {"content": user_message}
        if gemini_tools:
            kwargs["tools"] = gemini_tools

        response = chat.send_message(**kwargs)

        # Handle tool calls in a loop
        while response.candidates[0].content.parts:
            has_function_call = False
            function_responses = []

            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    has_function_call = True
                    fc = part.function_call
                    handler = tool_handlers.get(fc.name)

                    if handler:
                        try:
                            # Convert args to dict
                            args = dict(fc.args) if fc.args else {}
                            result = handler(**args)
                            function_responses.append({
                                "name": fc.name,
                                "response": {"result": str(result)},
                            })
                        except Exception as e:
                            function_responses.append({
                                "name": fc.name,
                                "response": {"error": str(e)},
                            })
                    else:
                        function_responses.append({
                            "name": fc.name,
                            "response": {"error": f"Unknown tool: {fc.name}"},
                        })

            if not has_function_call:
                break

            # Send function responses back
            # noinspection PyUnresolvedReferences
            from google.generativeai.types import Part

            response_parts = [
                Part.from_function_response(name=fr["name"], response=fr["response"])
                for fr in function_responses
            ]

            response = chat.send_message(response_parts, tools=gemini_tools)

        # Extract final text response
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        return "\n".join(text_parts)
