"""Configuration loading with Pydantic validation and env var substitution."""

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    name: str
    uri: str  # SQLAlchemy URI (env vars already substituted)


class LLMTiersConfig(BaseModel):
    """Model tiering for cost optimization."""
    planning: str = "claude-sonnet-4-20250514"
    codegen: str = "claude-sonnet-4-20250514"
    simple: str = "claude-3-5-haiku-20241022"


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    tiers: Optional[LLMTiersConfig] = None


class ExecutionConfig(BaseModel):
    """Execution settings for generated code."""
    timeout_seconds: int = 60
    max_retries: int = 10
    allowed_imports: list[str] = Field(default_factory=list)


class Config(BaseModel):
    """Root configuration model."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    databases: list[DatabaseConfig] = Field(default_factory=list)
    system_prompt: str = ""
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML file with env var substitution."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw_content = f.read()

        # Substitute environment variables: ${VAR_NAME}
        substituted = _substitute_env_vars(raw_content)

        data = yaml.safe_load(substituted)
        return cls.model_validate(data)


def _substitute_env_vars(content: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""
    pattern = re.compile(r'\$\{([^}]+)\}')

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Environment variable not set: {var_name}")
        return value

    return pattern.sub(replacer, content)
