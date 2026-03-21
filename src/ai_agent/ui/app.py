from __future__ import annotations

import os
import sys
import logging
import logging.handlers
from typing import List, Dict
from datetime import datetime

from dotenv import load_dotenv, find_dotenv

# Ensure ai_agent is in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load environment
load_dotenv(find_dotenv(), override=False)

log = logging.getLogger("chat_app")

from ai_agent.api.pipeline import RAGImagingPipeline
from ai_agent.core.pipeline_registry import get_pipeline as get_shared_pipeline
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.ui.components import create_chat_interface

# ============================================================================
# Tool registration
# ============================================================================
from ai_agent.agent.tools import ensure_tools_registered

# ============================================================================
# Pipeline initialization
# ============================================================================
INDEX_DIR = os.getenv("RAG_INDEX_DIR", "artifacts/rag_index")

_pipe: RAGImagingPipeline | None = None
_DOCS: List[SoftwareDoc] = []
_DOC_BY_NAME: Dict[str, SoftwareDoc] = {}
_logging_initialized = False
_tools_registered = False


def _setup_logging() -> None:
    """Initialize logging once for the UI process."""
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
    log.info("Starting Chat-based Gradio UI")


def ensure_logging_initialized() -> None:
    """Public hook for callers that need logging before other UI setup."""
    _setup_logging()


def _ensure_tools_registered_once() -> None:
    global _tools_registered
    if _tools_registered:
        return
    ensure_tools_registered()
    _tools_registered = True


def get_pipeline() -> RAGImagingPipeline:
    """Get or initialize the pipeline singleton."""
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


def refresh_ui_docs_from_index():
    """Refresh doc index from FAISS."""
    global _pipe, _DOCS, _DOC_BY_NAME
    if _pipe is None:
        return
    try:
        _DOCS = list(_pipe.index.docs.values())
        _DOC_BY_NAME = {d.name: d for d in _DOCS if getattr(d, "name", None)}
        log.info("UI docs refreshed from FAISS: %d tools", len(_DOCS))
    except Exception:
        _DOCS, _DOC_BY_NAME = [], {}
        log.exception("Failed to refresh UI docs from FAISS")


# ============================================================================
# Launch configuration
# ============================================================================
def _bind_host() -> str:
    """Determine bind host based on environment."""
    bind_host = os.getenv("BIND_HOST")
    if bind_host:
        return bind_host
    in_docker = os.path.exists("/.dockerenv")
    return "0.0.0.0" if in_docker else "127.0.0.1"


def launch():
    """Launch the chat interface."""
    _setup_logging()
    _ensure_tools_registered_once()
    host = _bind_host()
    preferred = int(os.getenv("PORT", "7860"))
    max_tries = int(os.getenv("PORT_TRIES", "10"))
    allow_fallback = str(os.getenv("ALLOW_PORT_FALLBACK", "1")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    ui = create_chat_interface(_DOC_BY_NAME)

    last_err = None
    for attempt in range(max_tries if allow_fallback else 1):
        port = preferred + attempt
        try:
            ui.queue(max_size=10).launch(
                server_name=host,
                server_port=port,
                inbrowser=False,
                show_error=True,
                share=bool(os.getenv("SHARE", False)),
            )
            if attempt > 0:
                log.info(
                    "Launched on fallback port %d (preferred %d was busy)",
                    port,
                    preferred,
                )
            return
        except OSError as e:
            last_err = e
            busy = "Cannot find empty port" in str(e)
            if not busy or attempt == (max_tries - 1) or not allow_fallback:
                raise
            log.warning("Port %d busy; trying %d", port, port + 1)

    if last_err:
        raise last_err


if __name__ == "__main__":
    launch()
