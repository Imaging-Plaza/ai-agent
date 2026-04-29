"""Utility functions for UI configuration management."""
import logging
from typing import Dict, Optional

from ai_agent.utils.config import get_config, get_available_models_config

log = logging.getLogger("ui.utils")


def get_agent_model() -> Dict[str, Optional[str]]:
    """
    Get the default agent model configuration from config.yaml.
    
    Returns:
        Dictionary with 'name', 'base_url', and 'api_key_env' keys.
        Returns minimal default if config not available.
    """
    agent_model = get_config().agent_model
    
    return {
        "name": agent_model.name,
        "base_url": agent_model.base_url,
        "api_key_env": agent_model.api_key_env,
    }


def get_available_models() -> Dict[str, Dict[str, Optional[str]]]:
    """
    Get available models from config.yaml for UI dropdown.
    
    Returns:
        Dictionary mapping display_name -> model config
        Returns minimal default if config not available.
    """
    available_models = get_available_models_config()
    
    model_configs = {}
    for model in available_models:
        display_name = model.get("display_name")
        if display_name:
            model_configs[display_name] = {
                "name": model.get("name"),
                "base_url": model.get("base_url"),
                "provider": model.get("provider", "Unknown"),
                "api_key_env": model.get("api_key_env", "OPENAI_API_KEY"),
            }
    
    # Fallback to minimal default if no models found
    if not model_configs:
        log.warning("No models found in config.yaml, using fallback")
        agent_model = get_agent_model()
        model_name = agent_model["name"]
        model_configs[model_name] = {
            "name": model_name,
            "base_url": agent_model["base_url"],
            "provider": "OpenAI",
            "api_key_env": agent_model["api_key_env"],
        }
    
    log.info(f"Loaded {len(model_configs)} models from config: {list(model_configs.keys())}")
    return model_configs


def get_default_model_display_name() -> str:
    """
    Get the display name for the default agent model.
    
    Returns:
        Display name string that matches one of the available_models,
        or the agent_model name if not found in available_models.
    """
    agent_model = get_agent_model()
    agent_name = agent_model["name"]
    available = get_available_models()
    
    # Try to find matching display name in available models
    for display_name, config in available.items():
        if config["name"] == agent_name:
            return display_name
    
    # Fallback: use agent name directly
    log.warning(f"Agent model '{agent_name}' not found in available_models, using as-is")
    return agent_name
