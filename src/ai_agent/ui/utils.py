"""Backward-compatible re-exports – canonical definitions live in core.model_config."""

from ai_agent.core.model_config import (  # noqa: F401
    get_agent_model,
    get_available_models,
    get_default_model_display_name,
    get_model_config,
)
