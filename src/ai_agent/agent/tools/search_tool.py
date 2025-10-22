from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

from generator.schema import CandidateDoc
from .utils import get_pipeline

class SearchToolsInput(BaseModel):
    query: str
    excluded: List[str] = Field(default_factory=list)
    top_k: int = 12
    original_formats: List[str] = Field(default_factory=list)

class SearchToolsOutput(BaseModel):
    candidates: List[CandidateDoc]

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