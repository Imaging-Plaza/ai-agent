"""Wire-format schemas for the FastAPI surface.

These pydantic models are mirrors of ``ai_agent.services.chat`` dataclasses
plus a couple of HTTP-shaped envelopes (login, session creation). They're
kept in their own module so they can be reused by:

  - the OpenAPI schema that ``openapi-typescript`` consumes in
    ``src/frontend`` to generate type-safe API clients;
  - any future Python integration test that hits the API directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    ok: bool = True


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
class SessionCreateResponse(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
class AssetResponse(BaseModel):
    asset_id: str
    display_name: Optional[str] = None
    original_format: Optional[str] = None
    preview_url: Optional[str] = None
    metadata_text: Optional[str] = None


class FilesUploadResponse(BaseModel):
    session_id: str
    assets: List[AssetResponse]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
class ChatStartBody(BaseModel):
    session_id: Optional[str] = None
    message: str = ""
    asset_ids: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    top_k: Optional[int] = None
    num_choices: Optional[int] = None


class RecommendationOut(BaseModel):
    rank: int
    name: str
    accuracy: float
    why: str
    doc: Optional[Dict[str, Any]] = None
    demo_url: Optional[str] = None


class PendingActionOut(BaseModel):
    type: Literal["demo_confirm", "tool_approval"]
    tool_name: str
    display_name: Optional[str] = None
    icon: Optional[str] = None
    image_name: Optional[str] = None
    demo_url: Optional[str] = None
    prompt: str = ""


class ClarificationOut(BaseModel):
    question: str
    context: Optional[str] = None
    options: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Models / catalog
# ---------------------------------------------------------------------------
class ModelOption(BaseModel):
    display_name: str
    name: str
    provider: Optional[str] = None


class CatalogEntry(BaseModel):
    name: str
    description: Optional[str] = ""
    modality: List[str] = Field(default_factory=list)
    license: Optional[str] = None
    dims: List[int] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    ok: bool = True
    catalog_docs: int = 0
    sessions: int = 0
