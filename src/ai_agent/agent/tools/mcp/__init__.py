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
    """
    Import all MCP tools to trigger their registration.
    Call this once at app startup.
    """
    from importlib import import_module

    import_module("ai_agent.agent.tools.mcp.lungs_segmentation_tool")
