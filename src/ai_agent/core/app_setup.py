"""Frontend-agnostic application setup: logging, pipeline, doc index, tool registration."""

from __future__ import annotations

import os
import sys
import logging
import logging.handlers
from typing import TYPE_CHECKING, List, Dict
from datetime import datetime

from ai_agent.core.pipeline_registry import get_pipeline as get_shared_pipeline

if TYPE_CHECKING:
    from ai_agent.api.pipeline import RAGImagingPipeline
    from ai_agent.retriever.software_doc import SoftwareDoc

log = logging.getLogger("app_setup")

# ============================================================================
# Shared state
# ============================================================================
INDEX_DIR = os.getenv("RAG_INDEX_DIR", "artifacts/rag_index")

_pipe: RAGImagingPipeline | None = None
_DOCS: List[SoftwareDoc] = []
_DOC_BY_NAME: Dict[str, SoftwareDoc] = {}
_logging_initialized = False
_tools_registered = False


# ============================================================================
# Logging
# ============================================================================
def _setup_logging() -> None:
    """Initialize logging once for the application process."""
    global _logging_initialized
    if _logging_initialized:
        return

    LOG_DIR = os.getenv("LOG_DIR", "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    debug_on = str(os.getenv("DEBUG", "0")).lower() in ("1", "true", "yes", "on")
    console_level = os.getenv("LOGLEVEL_CONSOLE", "INFO").upper()
    file_level = os.getenv("LOGLEVEL_FILE", "DEBUG" if debug_on else "INFO").upper()
    file_log_enabled = str(os.getenv("FILE_LOG", "1")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if file_log_enabled:
        logfile = os.path.join(LOG_DIR, f"app_{datetime.now():%Y%m%d}.log")
        fh = logging.handlers.TimedRotatingFileHandler(
            logfile, when="midnight", backupCount=14, encoding="utf-8"
        )
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _logging_initialized = True
    log.info("Logging initialized")


def ensure_logging_initialized() -> None:
    """Public hook for callers that need logging before other setup."""
    _setup_logging()


# ============================================================================
# Tool registration
# ============================================================================
def ensure_tools_registered_once() -> None:
    """Register agent tools exactly once."""
    global _tools_registered
    if _tools_registered:
        return
    from ai_agent.agent.tools import ensure_tools_registered  # lazy: heavy ML deps
    ensure_tools_registered()
    _tools_registered = True


# ============================================================================
# Pipeline & doc index
# ============================================================================
def get_pipeline() -> RAGImagingPipeline:
    """Get or initialize the pipeline singleton and populate doc index."""
    global _pipe, _DOCS, _DOC_BY_NAME
    if _pipe is None:
        _pipe = get_shared_pipeline(index_dir=INDEX_DIR)
        try:
            _DOCS = list(_pipe.index.docs.values())
            _DOC_BY_NAME = {d.name: d for d in _DOCS if getattr(d, "name", None)}
            log.info("Loaded %d tools from index", len(_DOCS))
        except Exception:
            _DOCS, _DOC_BY_NAME = [], {}
            log.exception("Failed to load docs from index")
        log.info("Pipeline ready")
    return _pipe


def get_doc_index() -> Dict[str, SoftwareDoc]:
    """Return the current doc-by-name index (populated after get_pipeline)."""
    return _DOC_BY_NAME


def refresh_docs_from_index() -> None:
    """Refresh doc index from FAISS."""
    global _pipe, _DOCS, _DOC_BY_NAME
    if _pipe is None:
        return
    try:
        _DOCS = list(_pipe.index.docs.values())
        _DOC_BY_NAME = {d.name: d for d in _DOCS if getattr(d, "name", None)}
        log.info("Docs refreshed from FAISS: %d tools", len(_DOCS))
    except Exception:
        _DOCS, _DOC_BY_NAME = [], {}
        log.exception("Failed to refresh docs from FAISS")
