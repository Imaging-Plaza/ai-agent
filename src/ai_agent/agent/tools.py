from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os, logging, re, json

from retriever.embedders import SoftwareDoc
from generator.schema import CandidateDoc
from utils.tags import strip_tags
from api.pipeline import RAGImagingPipeline  # reuse existing retrieval infra

log = logging.getLogger("agent.tools")

# We will lazily acquire the global pipeline created by UI or fallback build
_PIPE: Optional[RAGImagingPipeline] = None
_DOCS: List[SoftwareDoc] = []

class SearchToolsInput(BaseModel):
    query: str
    excluded: List[str] = Field(default_factory=list)
    top_k: int = 12
    original_formats: List[str] = Field(default_factory=list)

class SearchToolsOutput(BaseModel):
    candidates: List[CandidateDoc]


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
        _PIPE = RAGImagingPipeline(docs=docs)
    return _PIPE


def tool_search_tools(inp: SearchToolsInput) -> SearchToolsOutput:
    """Search tools WITHOUT reranker.
    Prefer explicit inp.original_formats. For backward compatibility, also
    supports legacy embedded line 'OriginalFormats:' inside the query.
    We append lightweight retrieval tokens (format:<EXT>) but DO NOT let them
    dominate semantics: they are only appended (not replacing content).
    """
    pipe = get_pipeline()
    q = inp.query
    original_formats: List[str] = [f.lower() for f in inp.original_formats]
    if not original_formats:
        for line in q.splitlines():
            if line.lower().startswith("originalformats:"):
                parts = line.split(":", 1)[1].strip().split()
                for p in parts:
                    ext = p.strip().lower()
                    if ext and ext not in original_formats:
                        original_formats.append(ext)
    # Remove any OriginalFormats line from semantic part
    clean_lines = [ln for ln in q.splitlines() if not ln.lower().startswith("originalformats:")]
    base_query = " ".join(ln.strip() for ln in clean_lines if ln.strip())
    # Build format tokens (uppercase canonical where useful)
    token_map = {
        'tif': 'TIFF', 'tiff': 'TIFF', 'nii': 'NIfTI', 'nii.gz': 'NIfTI', 'dcm': 'DICOM', 'dicom': 'DICOM',
        'nrrd': 'NRRD', 'png': 'PNG', 'jpg': 'JPEG', 'jpeg': 'JPEG'
    }
    fmt_tokens = []
    for ext in original_formats:
        canon = token_map.get(ext.lower(), ext.upper())
        if canon not in fmt_tokens:
            fmt_tokens.append(canon)
    if fmt_tokens:
        # append softly at end so primary semantics still dominate
        base_query = (base_query + " " + " ".join(f"format:{t}" for t in fmt_tokens)).strip()
    hits = pipe.retrieve_no_rerank(base_query, exclusions=inp.excluded, top_k=inp.top_k)
    cands: List[CandidateDoc] = []
    for h in hits:
        d = h.get("doc")
        if not d:
            continue
        try:
            cands.append(CandidateDoc.model_validate(d.model_dump(mode="python")))
        except Exception:
            continue
    return SearchToolsOutput(candidates=cands)

# -------------------- Rerank tool -------------------------------------------
class RerankInput(BaseModel):
    query: str
    candidate_names: List[str]
    top_k: int = int(os.getenv("TOP_K", "5"))

class RerankOutput(BaseModel):
    reranked: List[Dict[str, Any]]
    used_model: bool

def tool_rerank(inp: RerankInput) -> RerankOutput:
    pipe = get_pipeline()
    # reconstruct minimal hit dicts for reranker from catalog
    hits: List[Dict[str, Any]] = []
    for name in inp.candidate_names:
        doc = pipe.get_doc(name)
        if not doc:
            continue
        hits.append({"id": name, "doc": doc, "score": 0.0})
    if not hits:
        return RerankOutput(reranked=[], used_model=False)
    if getattr(pipe, "reranker", None):
        ranked = pipe.rerank_only(inp.query, hits, top_k=inp.top_k)
        out = [
            {
                "name": h["doc"].name,
                "score": float(h.get("rerank_score") or h.get("score") or 0.0),
            }
            for h in ranked
        ]
        return RerankOutput(reranked=out, used_model=True)
    # fallback lexical
    q = inp.query.lower()
    scored = []
    for h in hits:
        doc: SoftwareDoc = h["doc"]
        text = " ".join(filter(None, [doc.name, " ".join(doc.tasks), doc.description or ""])).lower()
        score = 0.0
        for tok in set(re.findall(r"[a-z0-9]+", q)):
            if tok in text:
                score += 1.0
        scored.append((doc.name, score))
    scored.sort(key=lambda x: -x[1])
    return RerankOutput(reranked=[{"name": n, "score": s} for n, s in scored[: inp.top_k]], used_model=False)


# Placeholder for run_example tool (Phase 2/3) ---------------------------------
# class RunExampleInput(BaseModel):
#     tool_name: str
#     image_path: Optional[str] = None  # local preview path
#     endpoint_url: Optional[str] = None  # override / explicit gradio space URL

# class RunExampleOutput(BaseModel):
#     tool_name: str
#     ran: bool = False
#     stdout: str = ""
#     generated_preview: Optional[str] = None
#     notes: Optional[str] = None
#     endpoint_url: Optional[str] = None


# def tool_run_example(inp: RunExampleInput) -> RunExampleOutput:
#     """Attempt to invoke a remote Gradio Space / endpoint using gradio_client if available.
#     Heuristic: use endpoint_url param else try to derive from catalog (if repo_url or runnable examples contain hf.space style links).
#     """
#     # lazy import gradio_client
#     try:
#         from gradio_client import Client
#     except Exception:
#         return RunExampleOutput(tool_name=inp.tool_name, ran=False, notes="gradio_client not installed")

#     pipe = get_pipeline()
#     doc = pipe.get_doc(inp.tool_name)
#     url = inp.endpoint_url
#     if not url and doc:
#         # search runnable examples inside raw doc model_dump
#         raw = doc.model_dump(mode="python", exclude_none=True)
#         examples = raw.get("runnableExample") or raw.get("runnable_example") or []
#         for ex in examples:
#             if isinstance(ex, dict):
#                 u = ex.get("url") or ""
#                 if "huggingface.co/spaces" in u or u.startswith("https://hf.space"):
#                     url = u
#                     break
#     if not url:
#         return RunExampleOutput(tool_name=inp.tool_name, ran=False, notes="No runnable example URL found")
#     # Basic call (no image upload yet)
#     try:
#         client = Client(url)
#         # We attempt to infer first predict endpoint; otherwise we just ping
#         stdout = ""
#         try:
#             apis = client.view_api(return_format="dict")
#             # naive choose first function that has no required image input if we have none
#             fn = None
#             for f in apis.get("endpoints", []):
#                 inputs = f.get("inputs", [])
#                 # If we have image_path, allow image input; else prefer text-only
#                 if not inp.image_path and any(i.get("type") == "image" for i in inputs):
#                     continue
#                 fn = f
#                 break
#             if fn:
#                 # build dummy inputs matching count (empty strings)
#                 input_count = len(fn.get("inputs", []))
#                 payload = [inp.tool_name] + ["" for _ in range(max(0, input_count - 1))]
#                 res = client.predict(*payload, api_name=fn.get("api_name"))
#                 stdout = str(res)
#             else:
#                 stdout = "No suitable predict endpoint discovered"
#         except Exception as e:
#             stdout = f"API introspection failed: {e}"
#         return RunExampleOutput(tool_name=inp.tool_name, ran=True, stdout=stdout[:4000], endpoint_url=url)
#     except Exception as e:
#         return RunExampleOutput(tool_name=inp.tool_name, ran=False, notes=str(e), endpoint_url=url)

# -------------------- Repo/Text enrichment tool -----------------------------
class RepoInfoInput(BaseModel):
    url: str
    max_chars: int = 4000

class RepoInfoOutput(BaseModel):
    url: str
    content: str
    truncated: bool = False

def tool_repo_info(inp: RepoInfoInput) -> RepoInfoOutput:
    """Very lightweight fallback to fetch README or index page. For GitHub repo URLs,
    attempt raw README retrieval. This can be replaced later with repo-to-text embedding service.
    """
    import requests
    url = inp.url
    text = ""
    try:
        if "github.com" in url and not url.endswith(".md"):
            parts = url.rstrip("/").split("/")
            if len(parts) >= 2:
                # derive user/repo
                user = parts[-2]
                repo = parts[-1]
                candidate = f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md"
                r = requests.get(candidate, timeout=10)
                if r.status_code == 200 and len(r.text) > 200:
                    text = r.text
        if not text:
            r = requests.get(url, timeout=10)
            text = r.text
    except Exception as e:
        return RepoInfoOutput(url=url, content=f"ERROR: {e}")
    trunc = False
    if len(text) > inp.max_chars:
        text = text[: inp.max_chars]
        trunc = True
    return RepoInfoOutput(url=url, content=text, truncated=trunc)

__all__ = [
    "tool_search_tools", "tool_rerank", "tool_run_example", "tool_fetch_url", "tool_repo_info",
    "SearchToolsInput", "SearchToolsOutput", "RerankInput", "RerankOutput",
    "RunExampleInput", "RunExampleOutput",
    "FetchUrlInput", "FetchUrlOutput", "RepoInfoInput", "RepoInfoOutput",
]
