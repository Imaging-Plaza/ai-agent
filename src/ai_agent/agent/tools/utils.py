from __future__ import annotations

from typing import List, Optional, Tuple
import os, json

from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.api.pipeline import RAGImagingPipeline


_PIPE: Optional[RAGImagingPipeline] = None
_DOCS: List[SoftwareDoc] = []
MAX_CHARS = 20000

def get_pipeline() -> RAGImagingPipeline:
    global _PIPE, _DOCS
    if _PIPE is None:
        # Minimal lazy loader; catalog path should already be set
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
                    line=line.strip()
                    if not line: continue
                    try:
                        docs.append(SoftwareDoc.model_validate(json.loads(line)))
                    except Exception:
                        continue
        _DOCS = docs
        _PIPE = RAGImagingPipeline()
    return _PIPE

def _clip(s: str) -> Tuple[str, bool]:
    if not s:
        return s, False
    if len(s) <= MAX_CHARS:
        return s, False
    return s[:MAX_CHARS] + "\n\n...[truncated for token budget]...", True