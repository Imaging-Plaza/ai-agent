"""Shared FastAPI dependencies.

  - ``require_auth``: simple cookie-based gate using ``APP_PASSWORD`` env var.
    If ``APP_PASSWORD`` is empty/unset, auth is disabled entirely (dev mode).
  - ``get_pipeline``: returns the singleton ``RAGImagingPipeline`` (lazy-init,
    shared with the Gradio path via ``ai_agent.core.pipeline_registry``).
  - ``get_doc_index``: name -> SoftwareDoc, derived from the pipeline.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Dict, Optional

from fastapi import Cookie, Depends, HTTPException, status

from ai_agent.api.pipeline import RAGImagingPipeline
from ai_agent.core.pipeline_registry import get_pipeline as _shared_pipeline
from ai_agent.retriever.software_doc import SoftwareDoc

log = logging.getLogger("api.deps")

AUTH_COOKIE_NAME = "ai_agent_auth"
_AUTH_VERSION = "v1"


def _expected_cookie() -> Optional[str]:
    """The cookie value clients must present, or None if auth is disabled."""
    pw = os.getenv("APP_PASSWORD") or ""
    if not pw:
        return None
    return hmac.new(
        pw.encode("utf-8"),
        _AUTH_VERSION.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def make_cookie_value(password: str) -> str:
    """Compute the cookie value for a successful login."""
    return hmac.new(
        password.encode("utf-8"),
        _AUTH_VERSION.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_password(password: str) -> bool:
    expected = os.getenv("APP_PASSWORD") or ""
    if not expected:
        # Auth disabled — accept any password (login is then effectively a noop).
        return True
    return hmac.compare_digest(password, expected)


def auth_disabled() -> bool:
    return not (os.getenv("APP_PASSWORD") or "")


async def require_auth(
    ai_agent_auth: Optional[str] = Cookie(default=None, alias=AUTH_COOKIE_NAME),
) -> None:
    """FastAPI dependency: 401 if cookie missing or wrong (when auth on)."""
    expected = _expected_cookie()
    if expected is None:
        return  # auth disabled
    if not ai_agent_auth or not hmac.compare_digest(ai_agent_auth, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="not_authenticated"
        )


# ---------------------------------------------------------------------------
# Pipeline / doc index providers (re-export so routers don't import internals).
# ---------------------------------------------------------------------------
def get_pipeline() -> RAGImagingPipeline:
    index_dir = os.getenv("RAG_INDEX_DIR", "artifacts/rag_index")
    return _shared_pipeline(index_dir=index_dir)


def get_doc_index(
    pipe: RAGImagingPipeline = Depends(get_pipeline),
) -> Dict[str, SoftwareDoc]:
    try:
        docs = list(pipe.index.docs.values())
        return {d.name: d for d in docs if getattr(d, "name", None)}
    except Exception:
        log.exception("Failed to build doc_index from pipeline")
        return {}


__all__ = [
    "AUTH_COOKIE_NAME",
    "auth_disabled",
    "get_doc_index",
    "get_pipeline",
    "make_cookie_value",
    "require_auth",
    "verify_password",
]
