from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel
import os, logging

from retriever.embedders import SoftwareDoc
from generator.schema import CandidateDoc
from utils.previews import _build_preview_for_vlm
from utils.tags import strip_tags, parse_exclusions, has_no_rerank
from utils.image_meta import detect_ext_token
from api.pipeline import RAGImagingPipeline  # reuse existing retrieval infra

log = logging.getLogger("agent.tools")

# We will lazily acquire the global pipeline created by UI or fallback build
_PIPE: Optional[RAGImagingPipeline] = None
_DOCS: List[SoftwareDoc] = []

class SearchToolsInput(BaseModel):
    query: str
    excluded: List[str] = []
    top_k: int = 12

class SearchToolsOutput(BaseModel):
    candidates: List[CandidateDoc]


def get_pipeline() -> RAGImagingPipeline:
    global _PIPE, _DOCS
    if _PIPE is None:
        # Minimal lazy loader; catalog path should already be set
        from pathlib import Path
        import json
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
        _PIPE = RAGImagingPipeline(docs=docs)
    return _PIPE


def tool_search_tools(inp: SearchToolsInput) -> SearchToolsOutput:
    """Search the catalog for candidate tools. Exclusions applied before rerank. Output subset of fields."""
    pipe = get_pipeline()
    # Build synthetic retrieval query (support format tokens like pipeline does if user included them earlier)
    hits, _scores = pipe.recommend(strip_tags(inp.query), image_paths=None, top_k=inp.top_k, persisted_exclusions=inp.excluded)
    cands: List[CandidateDoc] = []
    for h in hits:
        try:
            d = h["doc"]
            cands.append(CandidateDoc.model_validate(d.model_dump(mode="python")))
        except Exception:
            pass
    return SearchToolsOutput(candidates=cands)


# Placeholder for run_example tool (Phase 2/3) ---------------------------------
class RunExampleInput(BaseModel):
    tool_name: str
    image_path: Optional[str] = None

class RunExampleOutput(BaseModel):
    tool_name: str
    ran: bool = False
    stdout: str = ""
    generated_preview: Optional[str] = None
    notes: Optional[str] = None


def tool_run_example(inp: RunExampleInput) -> RunExampleOutput:
    # Phase 1 stub: do nothing, just acknowledge.
    return RunExampleOutput(tool_name=inp.tool_name, ran=False, notes="Stub run_example (Phase 1)")


# Simple URL fetch -------------------------------------------------------------
class FetchUrlInput(BaseModel):
    url: str

class FetchUrlOutput(BaseModel):
    content: str
    truncated: bool = False


def tool_fetch_url(inp: FetchUrlInput) -> FetchUrlOutput:
    import requests
    try:
        resp = requests.get(inp.url, timeout=10)
        txt = resp.text
    except Exception as e:
        return FetchUrlOutput(content=f"ERROR: {e}", truncated=False)
    max_len = 4000
    if len(txt) > max_len:
        return FetchUrlOutput(content=txt[:max_len], truncated=True)
    return FetchUrlOutput(content=txt, truncated=False)

__all__ = [
    "tool_search_tools", "tool_run_example", "tool_fetch_url",
    "SearchToolsInput", "SearchToolsOutput",
    "RunExampleInput", "RunExampleOutput",
    "FetchUrlInput", "FetchUrlOutput",
]
