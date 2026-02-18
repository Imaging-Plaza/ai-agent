import yaml
from pathlib import Path
from typing import List, Dict


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_available_models() -> List[Dict[str, str]]:
    """Get available models from config."""
    config = load_config()
    return config.get("available_models", [])


def get_default_ui_model() -> str:
    """Get default UI model from config."""
    config = load_config()
    return config.get("default_ui_model", "gpt-5.1")


def get_model_config(model_display_name: str) -> Dict[str, str]:
    """
    Get model configuration from display name.
    
    Args:
        model_display_name: Display name of the model (e.g., "gpt-4o-mini")
    
    Returns:
        Model configuration dict with name, base_url, provider, api_key_env
    """
    models = get_available_models()
    for model in models:
        if model.get("display_name") == model_display_name:
            return model
    # Fallback for unknown models
    return {
        "name": model_display_name,
        "base_url": None,
        "provider": "Unknown",
        "api_key_env": "OPENAI_API_KEY"
    }
