# utils/config.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import yaml

log = logging.getLogger("config")


class ModelConfig(BaseModel):
    """Configuration for a single model provider."""

    name: str = Field(
        ...,
        description="Model name (e.g., 'gpt-4o', 'mistralai/Mistral-Small-3.2-24B-Instruct-2506')",
    )
    base_url: Optional[str] = Field(
        None, description="API base URL (None for default OpenAI endpoint)"
    )
    api_key_env: str = Field(
        "OPENAI_API_KEY", description="Environment variable name for API key"
    )

    def get_api_key(self) -> str:
        """Get API key from environment variable."""
        key = os.getenv(self.api_key_env)
        if not key:
            raise ValueError(
                f"API key not found in environment variable: {self.api_key_env}"
            )
        return key


class AppConfig(BaseModel):
    """Agent model configuration loaded from config.yaml."""

    agent_model: ModelConfig = Field(
        ...,
        description="Model used for pydantic-ai agent (main reasoning & tool selection)",
    )


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load agent model configuration from config.yaml.

    Args:
        config_path: Path to config.yaml file. If None, looks for CONFIG_PATH env var

    Returns:
        AppConfig instance with agent model configuration
    """
    # Determine config file path
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH")

    # Load model from YAML if file exists
    if config_path and Path(config_path).exists():
        try:
            log.info(f"Loading model config from: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if "agent_model" in data:
                    return AppConfig(agent_model=ModelConfig(**data["agent_model"]))
        except (yaml.YAMLError, ValueError) as e:
            log.error(f"Failed to load config from {config_path}: {e}")
            log.warning("Falling back to default configuration")

    # Fall back to default model
    log.warning(
        "No config.yaml found or no agent_model defined, using default model from environment"
    )
    return AppConfig(
        agent_model=ModelConfig(
            name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=None,
            api_key_env="OPENAI_API_KEY",
        )
    )


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance (loads on first access)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
