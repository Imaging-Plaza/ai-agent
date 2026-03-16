from __future__ import annotations

from typing import List, Optional, Tuple
import os
import json
from urllib.parse import urlparse

from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.core.pipeline_registry import get_pipeline as get_shared_pipeline
from ai_agent.api.pipeline import RAGImagingPipeline

_DOCS: List[SoftwareDoc] = []
MAX_CHARS = 20000


def get_catalog_docs() -> List[SoftwareDoc]:
    """
    Load and return catalog docs without initializing the full pipeline.

    This is a lightweight alternative to get_pipeline() for catalog-only operations
    that don't need the embedder, reranker, or index.
    """
    global _DOCS
    if not _DOCS:
        # Load catalog docs
        from pathlib import Path

        catalog = os.getenv("SOFTWARE_CATALOG", "data/sample.jsonl")
        p = Path(catalog)
        docs: List[SoftwareDoc] = []
        if p.exists():
            text = p.read_text(encoding="utf-8").strip()
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    obj = [obj]
                for o in obj:
                    docs.append(SoftwareDoc.model_validate(o))
            except Exception:
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        docs.append(SoftwareDoc.model_validate(json.loads(line)))
                    except Exception:
                        continue
        _DOCS = docs
    return _DOCS


def get_pipeline() -> "RAGImagingPipeline":
    # Load docs first (reuses cached docs if available)
    get_catalog_docs()
    # Use the process-wide shared pipeline singleton.
    return get_shared_pipeline()


def _clip(s: str) -> Tuple[str, bool]:
    if not s:
        return s, False
    if len(s) <= MAX_CHARS:
        return s, False
    return s[:MAX_CHARS] + "\n\n...[truncated for token budget]...", True


def _is_github_url(url: str) -> bool:
    """Return True only for URLs that actually point to github.com."""
    s = (url or "").strip()
    if not s:
        return False

    parsed = urlparse(s)

    # If no scheme was provided (e.g. "github.com/org/repo"), parse again with a dummy scheme
    if not parsed.scheme and not parsed.netloc:
        parsed = urlparse("https://" + s)

    return parsed.netloc.lower() == "github.com"
