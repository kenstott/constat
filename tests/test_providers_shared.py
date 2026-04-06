from __future__ import annotations

# Copyright (c) 2025 Kenneth Stott
# Canary: f0e226dd-3488-41ae-80c7-40d2fbe56fa8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Shared constants and helpers used across provider test modules."""

# Sample tools for testing tool calling
SAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
]


def get_weather(location: str) -> str:
    """Mock weather tool handler."""
    return f"Weather in {location}: 72F, sunny"


def calculate(expression: str) -> str:
    """Mock calculator tool handler."""
    try:
        result = eval(expression)  # Safe for tests with controlled input
        return str(result)
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "get_weather": get_weather,
    "calculate": calculate,
}
