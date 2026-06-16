"""
MCP (Model Context Protocol) tools package.

This package contains registered imaging tools that require approval
and follow the tool registry pattern.
"""

from .registry import (
    TOOL_REGISTRY,
    CATALOG_NAME_TO_TOOL,
    get_tool,
    register_tool,
    list_tools,
    initialize_registry,
    reload_registry,
    resolve_catalog_alias,
    active_config_json,
    active_config_path,
    validate_config_payload,
    save_config_payload,
    GRADIO_TOOLS_CONFIG_ENV,
    RegistryValidationError,
    get_tool_display_name,
    get_tool_icon,
    extract_preview,
    extract_downloads,
    extract_metadata,
    extract_output_field,
    ToolConfig,
)

from .base import BaseToolInput, BaseToolOutput, ImageToolInput

__all__ = [
    # Registry
    "TOOL_REGISTRY",
    "CATALOG_NAME_TO_TOOL",
    "get_tool",
    "register_tool",
    "list_tools",
    "initialize_registry",
    "reload_registry",
    "resolve_catalog_alias",
    "active_config_json",
    "active_config_path",
    "validate_config_payload",
    "save_config_payload",
    "GRADIO_TOOLS_CONFIG_ENV",
    "RegistryValidationError",
    "get_tool_display_name",
    "get_tool_icon",
    "extract_preview",
    "extract_downloads",
    "extract_metadata",
    "extract_output_field",
    "ToolConfig",
    # Base models
    "BaseToolInput",
    "BaseToolOutput",
    "ImageToolInput",
]


def ensure_mcp_tools_registered():
    """Load configured Gradio tools into the shared registry."""
    initialize_registry()
