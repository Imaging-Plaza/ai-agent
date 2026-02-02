# generator/schema.py
from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class SupportingData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    content_url: Optional[str] = Field(default=None, alias="contentUrl")
    description: Optional[str] = None
    name: Optional[str] = None
    dataset_format: Optional[str] = Field(default=None, alias="datasetFormat")
    has_dimensionality: Optional[int] = Field(default=None, alias="hasDimensionality")
    body_site: Optional[str] = Field(default=None, alias="bodySite")
    imaging_modality: Optional[str] = Field(default=None, alias="imagingModality")

class RunnableExample(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    description: Optional[str] = None
    name: Optional[str] = None
    url: Optional[str] = None
    host_type: Optional[str] = Field(default=None, alias="hostType")

class ExecutableNotebook(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    description: Optional[str] = None
    name: Optional[str] = None
    url: Optional[str] = None

class CandidateDoc(BaseModel):
    """
    Minimal view of a software tool passed to the generator (aligned with SoftwareDoc).
    All fields optional/lenient to tolerate catalog variation.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # Core identity
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None

    # Categories & semantics
    category: List[str] = Field(default_factory=list, alias="applicationCategory")
    tasks: List[str] = Field(default_factory=list, alias="featureList")
    keywords: List[str] = Field(default_factory=list)

    # Modality / anatomy / dimensionality
    modality: List[str] = Field(default_factory=list, alias="imagingModality")
    dims: List[int] = Field(default_factory=list)            # normalized to ints (2,3,4)
    anatomy: List[str] = Field(default_factory=list)         # from supportingData[*].bodySite

    # Tech details
    programming_language: Optional[str] = Field(default=None, alias="programmingLanguage")
    software_requirements: List[str] = Field(default_factory=list, alias="softwareRequirements")
    gpu_required: Optional[bool] = Field(default=None, alias="requiresGPU")
    is_free: Optional[bool] = Field(default=None, alias="isAccessibleForFree")
    is_based_on: List[str] = Field(default_factory=list, alias="isBasedOn")
    plugin_of: List[str] = Field(default_factory=list, alias="isPluginModuleOf")  # <-- CHANGED to list
    related_organizations: List[str] = Field(default_factory=list, alias="relatedToOrganization")

    # Rich, nested metadata
    supporting_data: List[SupportingData] = Field(default_factory=list, alias="supportingData")
    runnable_examples: List[RunnableExample] = Field(default_factory=list, alias="runnableExample")
    executable_notebooks: List[ExecutableNotebook] = Field(default_factory=list, alias="hasExecutableNotebook")

    # -------- Normalizers --------
    @field_validator("plugin_of", mode="before")
    @classmethod
    def _coerce_plugin_of(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
        # single value -> list
        s = str(v).strip()
        return [s] if s else []

    @field_validator("dims", mode="before")
    @classmethod
    def _coerce_dims(cls, v):
        """
        Accepts: int/float, '3D'/'2-D', '3', 'volumetric','stack','timeseries', lists, or comma strings.
        Returns de-duped list[int].
        """
        if v is None:
            return []
        items = []
        if isinstance(v, list):
            items = v
        elif isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            items = parts or [v]
        else:
            items = [v]

        out: list[int] = []

        def push(x):
            try:
                xi = int(x)
                if xi not in out:
                    out.append(xi)
            except Exception:
                pass

        for it in items:
            if isinstance(it, (int, float)):
                push(it); continue
            if not isinstance(it, str):
                continue
            s = it.strip().lower().replace(" ", "")
            if s in {"2", "2d", "2-d"}: push(2); continue
            if s in {"3", "3d", "3-d", "volume", "volumetric", "stack"}: push(3); continue
            if s in {"4", "4d", "4-d", "timeseries", "time-series", "temporal"}: push(4); continue
            digits = "".join(ch for ch in s if ch.isdigit())
            if digits: push(digits)
        return out

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

class NoToolReason(str, Enum):
    NO_SUITABLE_TOOL = "no_suitable_tool"
    NO_MODALITY_MATCH = "no_modality_match"
    NO_TASK_MATCH = "no_task_match"
    NO_DIMENSION_MATCH = "no_dimension_match"
    INVALID_FILES = "invalid_files"  

class ConversationStatus(str, Enum):
    NEEDS_CLARIFICATION = "needs_clarification"
    COMPLETE = "complete"

class Conversation(BaseModel):
    status: ConversationStatus
    question: Optional[str] = None
    context: Optional[str] = None
    options: Optional[List[str]] = None

    @model_validator(mode='after')
    def validate_fields(self) -> 'Conversation':
        if self.status == ConversationStatus.NEEDS_CLARIFICATION:
            if not self.question:
                raise ValueError("Question required when status is needs_clarification")
            if not self.context:
                raise ValueError("Context required when status is needs_clarification")
        return self

class ToolChoice(BaseModel):
    name: str
    rank: int
    accuracy: float = Field(ge=0, le=100)  # accuracy score between 0-100
    why: str
    demo_link: Optional[str] = None

class ToolSelection(BaseModel):
    conversation: Conversation
    choices: List[ToolChoice] = Field(default_factory=list)
    explanation: Optional[str] = None
    reason: Optional[NoToolReason] = None

    @model_validator(mode="after")
    def normalize(self) -> "ToolSelection":
        has_choices = len(self.choices) > 0
        asked_q = bool(self.conversation.question and str(self.conversation.question).strip())

        if asked_q:
            # Clarification turn
            self.conversation.status = ConversationStatus.NEEDS_CLARIFICATION
            self.reason = None
            self.explanation = None
            return self

        if has_choices:
            # Normal success: per-tool 'why' explains rationale; no global reason
            self.conversation.status = ConversationStatus.COMPLETE
            self.reason = None
            # explanation optional but usually unnecessary; keep if you want, or null it:
            # self.explanation = None
            return self

        # No choices
        if self.reason or (self.explanation and self.explanation.strip()):
            # Terminal no-tool
            self.conversation.status = ConversationStatus.COMPLETE
            return self

        # Ambiguous empty -> ask a question
        self.conversation.status = ConversationStatus.NEEDS_CLARIFICATION
        return self

    @field_validator("reason", mode="before")
    def _reason_empty_to_none(cls, v):
        return None if v is None or str(v).strip() == "" else v

    @field_validator("explanation", mode="before")
    def _expl_empty_to_none(cls, v):
        return None if v is None or str(v).strip() == "" else v


__all__ = [
    "CandidateDoc",
    "PlanAndCode",
    "ToolSelection",
    "Conversation",
    "ConversationStatus",
    "ToolChoice",
    "NoToolReason",
]