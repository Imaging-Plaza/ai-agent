"""Liveness probe + small operational counters."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ai_agent.api.deps import get_pipeline
from ai_agent.api.pipeline import RAGImagingPipeline
from ai_agent.api.schemas import HealthResponse
from ai_agent.services.sessions import get_session_store

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz(pipe: RAGImagingPipeline = Depends(get_pipeline)) -> HealthResponse:
    try:
        docs = len(pipe.index.docs)
    except Exception:
        docs = 0
    return HealthResponse(ok=True, catalog_docs=docs, sessions=get_session_store().count())
