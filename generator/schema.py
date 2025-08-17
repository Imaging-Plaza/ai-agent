# generator/schema.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PerceptionCues(BaseModel):
    """Lightweight cues inferred from the image/text to guide retrieval/selection."""
    modality: Optional[str] = None   # e.g., CT, MRI, XR, US, natural
    dims: Optional[str] = None       # e.g., 2D, 3D, 4D, unknown
    anatomy: Optional[str] = None    # e.g., lung, heart
    task: Optional[str] = None       # e.g., segmentation, deblurring, registration
    io_hint: Optional[str] = None    # e.g., NIfTI, TIFF


class CandidateDoc(BaseModel):
    """
    Minimal view of a software tool passed to the generator.
    Keep fields optional so catalog variations don't break validation.
    """
    name: str
    tasks: List[str] = Field(default_factory=list)
    modality: List[str] = Field(default_factory=list)
    dims: List[str] = Field(default_factory=list)
    anatomy: List[str] = Field(default_factory=list)

    input_formats: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)

    language: Optional[str] = None
    weights_available: Optional[bool] = None
    gpu_required: Optional[bool] = None
    os: List[str] = Field(default_factory=list)
    license: Optional[str] = None
    description: Optional[str] = None

    # Optional demo metadata (safe to include even if unused)
    hf_space: Optional[str] = None
    hf_api_name: Optional[str] = None
    space_timeout: Optional[int] = None


class PlanAndCode(BaseModel):
    """
    Back-compat schema for the older 'plan + code' generator.
    Your pipeline can ignore steps/code, but we keep it so imports don't fail.
    """
    choice: str
    alternates: List[str] = Field(default_factory=list)
    why: str
    steps: List[str] = Field(default_factory=list)
    code: str = ""


class ToolSelection(BaseModel):
    """
    New, minimal schema used by the selection-only generator.
    """
    choice: str
    alternates: List[str] = Field(default_factory=list)
    why: str


__all__ = [
    "PerceptionCues",
    "CandidateDoc",
    "PlanAndCode",
    "ToolSelection",
]
