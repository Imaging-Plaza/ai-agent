from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

from ai_agent.generator.schema import CandidateDoc
from .utils import get_pipeline

class SearchToolsInput(BaseModel):
    query: str
    excluded: List[str] = Field(default_factory=list)
    top_k: int = 12
    original_formats: List[str] = Field(default_factory=list)
    image_paths: List[str] = Field(default_factory=list)

class SearchToolsOutput(BaseModel):
    candidates: List[CandidateDoc]

def tool_search_tools(inp: SearchToolsInput) -> SearchToolsOutput:
    """
    Search tools WITHOUT reranker.

    - Uses dense retrieval with similarity-based query expansion.
    - Softly biases results using file-format hints (format:EXT).
    - Optionally uses `image_paths` so the pipeline can derive additional
      hints (modality / anatomy / dims) directly from the image files.
    - Includes automatic retry logic if insufficient results are found.
    """
    pipe = get_pipeline()

    # 1) Start from the raw query
    q = inp.query

    # 2) Normalise original formats
    original_formats: List[str] = [f.lower() for f in inp.original_formats]

    # If none were explicitly provided, look for a legacy "OriginalFormats:" line.
    if not original_formats:
        for line in q.splitlines():
            if line.lower().startswith("originalformats:"):
                parts = line.split(":", 1)[1].strip().split()
                for p in parts:
                    ext = p.strip().lower()
                    if ext and ext not in original_formats:
                        original_formats.append(ext)

    # 3) Remove any "OriginalFormats:" line from the semantic query
    clean_lines = [
        ln for ln in q.splitlines()
        if not ln.lower().startswith("originalformats:")
    ]
    base_query = " ".join(ln.strip() for ln in clean_lines if ln.strip())

    # 4) Build soft format tokens (they bias but do not dominate)
    token_map = {
        "tif": "TIFF",
        "tiff": "TIFF",
        "nii": "NIfTI",
        "nii.gz": "NIfTI",
        "dcm": "DICOM",
        "dicom": "DICOM",
        "nrrd": "NRRD",
        "png": "PNG",
        "jpg": "JPEG",
        "jpeg": "JPEG",
    }
    fmt_tokens: List[str] = []
    for ext in original_formats:
        canon = token_map.get(ext.lower(), ext.upper())
        if canon not in fmt_tokens:
            fmt_tokens.append(canon)

    if fmt_tokens:
        # append softly at end so primary semantics still dominate
        base_query = (
            base_query + " " + " ".join(f"format:{t}" for t in fmt_tokens)
        ).strip()

    # 5) Call the vector index with similarity expansion and automatic retry
    # The pipeline now handles similarity-based expansion internally
    hits = pipe.retrieve_no_rerank(
        base_query,
        image_paths=inp.image_paths or None,
        exclusions=inp.excluded,
        top_k=inp.top_k,
    )

    # 6) Convert hits back into CandidateDoc objects for the agent
    candidates: List[CandidateDoc] = []
    for h in hits:
        d = h.get("doc")
        if not d:
            continue
        try:
            candidates.append(
                CandidateDoc.model_validate(d.model_dump(mode="python"))
            )
        except Exception:
            continue

    return SearchToolsOutput(candidates=candidates)