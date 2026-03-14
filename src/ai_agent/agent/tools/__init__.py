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
    from .search_tool import tool_search_tools
    from .search_alternative_tool import tool_search_alternative
    from .repo_info_tool import tool_repo_summary
    from .gradio_space_tool import tool_run_example

    # Import MCP tools
    ensure_mcp_tools_registered()
