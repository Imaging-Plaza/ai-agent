from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_agent.api.pipeline import RAGImagingPipeline

_PIPELINE: Optional["RAGImagingPipeline"] = None
_PIPELINE_INDEX_DIR: Optional[Path] = None
_PIPELINE_CLASS = None


def _resolve_index_dir(index_dir: Optional[str] = None) -> Path:
    return Path(index_dir or os.getenv("RAG_INDEX_DIR", "artifacts/rag_index")).resolve()


def get_pipeline(index_dir: Optional[str] = None) -> "RAGImagingPipeline":
    """Return the shared pipeline singleton across CLI, UI and tools."""
    global _PIPELINE, _PIPELINE_INDEX_DIR

    requested_dir = _resolve_index_dir(index_dir)
    pipeline_cls = _PIPELINE_CLASS
    if pipeline_cls is None:
        from ai_agent.api.pipeline import RAGImagingPipeline

        pipeline_cls = RAGImagingPipeline

    if _PIPELINE is None:
        _PIPELINE = pipeline_cls(index_dir=str(requested_dir))
        _PIPELINE_INDEX_DIR = requested_dir
    elif _PIPELINE_INDEX_DIR is not None and requested_dir != _PIPELINE_INDEX_DIR:
        # Recreate only when a different index dir is explicitly requested.
        _PIPELINE = pipeline_cls(index_dir=str(requested_dir))
        _PIPELINE_INDEX_DIR = requested_dir

    return _PIPELINE


def reset_pipeline() -> None:
    """Reset shared pipeline (used by tests)."""
    global _PIPELINE, _PIPELINE_INDEX_DIR, _PIPELINE_CLASS
    _PIPELINE = None
    _PIPELINE_INDEX_DIR = None
    _PIPELINE_CLASS = None
