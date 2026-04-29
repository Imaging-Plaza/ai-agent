from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class BaseToolInput(BaseModel):
    """
    Base input model that tools can extend.

    Common patterns:
    - image_path: Path to uploaded image/volume
    - description: Optional context from agent
    """

    pass  # Intentionally minimal - tools define their own inputs


class BaseToolOutput(BaseModel):
    """
    Base output model that all tools should follow.

    This ensures consistent handling in the UI layer without
    needing tool-specific code.

    Standard fields:
    - success: bool - Whether execution succeeded
    - error: Optional[str] - Error message if failed
    - compute_time_seconds: float - Time taken by tool
    - notes: Optional[str] - Additional info for user

    File outputs (at least one should be provided on success):
    - result_preview: Optional[str] - PNG/GIF preview for inline display
    - result_origin: Optional[str] - Original format file for download
    - result_path: Optional[str] - Backward compat field

    Metadata:
    - metadata_text: Optional[str] - Structured info about result
    - metadata: Dict[str, Any] - Machine-readable metadata

    Tracking:
    - endpoint_url: str - API endpoint used
    - api_name: str - API method called
    """

    # Core status
    success: bool = False
    error: Optional[str] = None
    compute_time_seconds: float = 0.0

    # File outputs (tools should provide these for UI to display/download)
    result_preview: Optional[str] = Field(
        default=None,
        description="Path to preview image (PNG/GIF) for inline display in chat",
    )
    result_origin: Optional[str] = Field(
        default=None,
        description="Path to original format file (TIFF/NIfTI/DICOM) for download",
    )
    result_path: Optional[str] = Field(
        default=None, description="Backward compatibility: primary result path"
    )

    # Metadata
    metadata_text: Optional[str] = Field(
        default=None, description="Human-readable metadata about the result"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Machine-readable metadata"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes or context for user"
    )

    # Tracking/debugging
    endpoint_url: str = ""
    api_name: str = ""


class ImageToolInput(BaseToolInput):
    """
    Common input pattern for image/volume processing tools.
    """

    image_path: str = Field(description="Path to the image/volume file")
    description: Optional[str] = Field(
        default=None, description="Optional context or notes from agent about the task"
    )
