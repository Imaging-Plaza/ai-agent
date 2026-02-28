from __future__ import annotations

from typing import Dict, Type, Callable, Optional, List, Any
from pydantic import BaseModel
from dataclasses import dataclass


@dataclass
class ToolConfig:
    """
    Declarative configuration for a tool.
    
    Tools register themselves with this config, and the UI uses it
    to generically handle execution, display, and file management.
    """
    # Core identification
    name: str  # Internal name (e.g., "lungs_segmentation")
    display_name: str  # User-facing name (e.g., "3D Lungs Segmentation")
    icon: str  # Emoji for UI display
    
    # Type information
    input_model: Type[BaseModel]  # Pydantic model for inputs
    output_model: Type[BaseModel]  # Pydantic model for outputs
    executor: Callable  # Function that takes input_model and returns output_model
    
    # Capability flags
    catalog_names: Optional[List[str]] = None  # Catalog names for this tool (e.g., ["lungs-segmentation"])
    supports_images: bool = True
    supports_files: bool = True
    requires_approval: bool = True  # Whether to show approval button
    
    # Output field mappings (how to extract results from output_model)
    # These map generic concepts to tool-specific field names
    preview_field: str = "result_preview"  # Field containing preview image path
    download_fields: List[str] | str = "result_origin"  # Field(s) for downloadable files
    metadata_field: Optional[str] = "metadata_text"  # Optional metadata text
    notes_field: str = "notes"  # Field containing execution notes
    
    # Success detection
    success_field: str = "success"  # Field indicating success/failure
    error_field: str = "error"  # Field containing error message
    compute_time_field: str = "compute_time_seconds"  # Field with timing info


# Global tool registry
TOOL_REGISTRY: Dict[str, ToolConfig] = {}

# Reverse mapping from catalog names to tool names
CATALOG_NAME_TO_TOOL: Dict[str, str] = {}


def register_tool(config: ToolConfig) -> None:
    """
    Register a tool with the global registry.
    
    Args:
        config: Tool configuration
        
    Raises:
        ValueError: If tool name already registered or catalog name collision
    """
    if config.name in TOOL_REGISTRY:
        raise ValueError(f"Tool '{config.name}' is already registered")
    
    # Check for catalog name collisions before registering
    if config.catalog_names:
        for catalog_name in config.catalog_names:
            if catalog_name in CATALOG_NAME_TO_TOOL and CATALOG_NAME_TO_TOOL[catalog_name] != config.name:
                raise ValueError(
                    f"Catalog name '{catalog_name}' already registered to "
                    f"'{CATALOG_NAME_TO_TOOL[catalog_name]}'"
                )
    
    TOOL_REGISTRY[config.name] = config
    
    # Register catalog name mappings
    if config.catalog_names:
        for catalog_name in config.catalog_names:
            CATALOG_NAME_TO_TOOL[catalog_name] = config.name


def get_tool(name: str) -> Optional[ToolConfig]:
    """
    Get tool configuration by name.
    
    Args:
        name: Tool name (registry name or catalog name)
        
    Returns:
        ToolConfig if found, None otherwise
    
    Note:
        This function checks both the tool registry name and catalog names.
    """
    # First try direct registry lookup
    config = TOOL_REGISTRY.get(name)
    if config:
        return config
    
    # Try catalog name mapping
    tool_name = CATALOG_NAME_TO_TOOL.get(name)
    if tool_name:
        return TOOL_REGISTRY.get(tool_name)
    
    return None


def list_tools() -> List[str]:
    """Get list of all registered tool names."""
    return list(TOOL_REGISTRY.keys())


def get_tool_display_name(name: str) -> str:
    """
    Get display name for a tool, with fallback to name.
    
    Args:
        name: Tool name
        
    Returns:
        Display name or formatted version of name
    """
    tool = get_tool(name)
    if tool:
        return tool.display_name
    # Fallback: format name nicely
    return name.replace("_", " ").title()


def get_tool_icon(name: str) -> str:
    """
    Get icon for a tool, with fallback.
    
    Args:
        name: Tool name
        
    Returns:
        Icon emoji or default
    """
    tool = get_tool(name)
    if tool:
        return tool.icon
    return "🔧"  # Default tool icon


def extract_output_field(output: BaseModel, field_name: str) -> Any:
    """
    Safely extract a field from tool output.
    
    Args:
        output: Tool output object
        field_name: Field name to extract
        
    Returns:
        Field value or None if not found
    """
    return getattr(output, field_name, None)


def extract_preview(output: BaseModel, tool_name: str) -> Optional[str]:
    """Extract preview image path from tool output."""
    tool = get_tool(tool_name)
    if not tool:
        return None
    return extract_output_field(output, tool.preview_field)


def extract_downloads(output: BaseModel, tool_name: str) -> List[str]:
    """Extract downloadable file paths from tool output."""
    tool = get_tool(tool_name)
    if not tool:
        return []
    
    download_fields = tool.download_fields
    if isinstance(download_fields, str):
        download_fields = [download_fields]
    
    downloads = []
    for field in download_fields:
        value = extract_output_field(output, field)
        if value:
            if isinstance(value, list):
                downloads.extend([v for v in value if v])
            elif isinstance(value, str):
                downloads.append(value)
    
    return [d for d in downloads if d]  # Filter None/empty


def extract_metadata(output: BaseModel, tool_name: str) -> Optional[str]:
    """Extract metadata text from tool output."""
    tool = get_tool(tool_name)
    if not tool or not tool.metadata_field:
        return None
    return extract_output_field(output, tool.metadata_field)
