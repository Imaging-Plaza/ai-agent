"""Agent tools package."""

# Only export registry - tools will self-register when imported explicitly
from .mcp import (
    TOOL_REGISTRY,
    get_tool,
    register_tool,
    list_tools,
    ensure_mcp_tools_registered,
)

# Import tools lazily to avoid loading heavy dependencies at package import
# Tools should be imported explicitly where needed, e.g.:
#   from ai_agent.agent.tools.mcp.lungs_segmentation_tool import tool_lungs_segmentation

__all__ = [
    "TOOL_REGISTRY",
    "get_tool",
    "register_tool",
    "list_tools",
    "ensure_tools_registered",
]


def ensure_tools_registered():
    """
    Import all tools to trigger their registration.
    Call this once at app startup.
    """
    from importlib import import_module

    import_module("ai_agent.agent.tools.search_tool")
    import_module("ai_agent.agent.tools.search_alternative_tool")
    import_module("ai_agent.agent.tools.repo_info_tool")
    import_module("ai_agent.agent.tools.gradio_space_tool")

    # Import MCP tools
    ensure_mcp_tools_registered()
