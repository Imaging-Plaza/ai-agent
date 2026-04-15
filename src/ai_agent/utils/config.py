# utils/config.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List
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

def _resolve_config_path(config_path: Optional[str] = None) -> Optional[Path]:
    path = config_path or os.getenv("CONFIG_PATH")
    if not path:
        return None
    p = Path(path)
    return p if p.exists() else None


def load_raw_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load raw YAML config dictionary from disk, returning {} on error/missing file."""
    p = _resolve_config_path(config_path)
    if not p:
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        log.error(f"Failed to load config from {p}: {e}")
        return {}


def get_available_models_config(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return available_models entries from config.yaml."""
    data = load_raw_config(config_path)
    models = data.get("available_models", [])
    return models if isinstance(models, list) else []


def get_retrieval_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return retrieval settings from config.yaml.

    Expected shape:
        retrieval:
            embedder: {...}
            reranker: {...}
    """
    data = load_raw_config(config_path)
    retrieval = data.get("retrieval", {})
    return retrieval if isinstance(retrieval, dict) else {}


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load agent model configuration from config.yaml.

    Args:
        config_path: Path to config.yaml file. If None, looks for CONFIG_PATH env var

    Returns:
        AppConfig instance with agent model configuration
    """
    data = load_raw_config(config_path)
    if data.get("agent_model"):
        try:
            return AppConfig(agent_model=ModelConfig(**data["agent_model"]))
        except ValueError as e:
            log.error(f"Invalid agent_model in config: {e}")
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


__all__ = [
    "ModelConfig",
    "AppConfig",
    "load_raw_config",
    "get_available_models_config",
    "get_retrieval_config",
    "load_config",
    "get_config",
]
