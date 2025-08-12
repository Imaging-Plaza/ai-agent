# generator/schema.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, validator

class PerceptionCues(BaseModel):
    modality: Optional[str] = None    # e.g., CT, MRI
    dims: Optional[str] = None        # "2D" | "3D"
    anatomy: Optional[str] = None     # lung, brain, ...
    task: Optional[str] = None        # segmentation, registration, ...
    io_hint: Optional[str] = None     # NIfTI, DICOM, PNG, ...

class CandidateDoc(BaseModel):
    name: str
    repo_url: Optional[str] = None
    tasks: List[str] = Field(default_factory=list)
    modality: List[str] = Field(default_factory=list)
    dims: List[str] = Field(default_factory=list)
    anatomy: List[str] = Field(default_factory=list)
    input_formats: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    install_cmd: Optional[str] = None
    weights_available: Optional[bool] = None
    license: Optional[str] = None
    gpu_required: Optional[bool] = None
    sample_snippet: Optional[str] = None

class PlanAndCode(BaseModel):
    choice: str
    alternates: List[str] = Field(default_factory=list, max_items=2)
    why: str
    steps: List[str] = Field(default_factory=list, max_items=4)
    code: str

    @validator("code")
    def short_enough(cls, v: str):
        if v.count("\n") > 40:
            raise ValueError("Code too long; keep under ~40 lines for demo.")
        return v
