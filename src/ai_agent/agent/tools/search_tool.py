from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

from ai_agent.generator.schema import CandidateDoc
from .utils import get_known_names, get_pipeline
from .query_utils import (
    append_format_tokens,
    normalize_formats,
    sanitize_retrieval_query,
    strip_legacy_original_formats_line,
)


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
    Search tools with automatic reranking.

    - Uses embedding-based similarity and metadata hints
    - Applies CrossEncoder reranking automatically for best results.
    - Softly biases results using file-format hints (format:EXT).
    - Optionally uses `image_paths` so the pipeline can derive additional
      hints (modality / anatomy / dims) directly from the image files.
    """
    pipe = get_pipeline()

    # 1) Start from the raw query
    q = inp.query

    # 2) Normalise original formats
    original_formats: List[str] = normalize_formats(inp.original_formats)

    # If none were explicitly provided, look for a legacy "OriginalFormats:" line.
    if not original_formats:
        _, from_legacy = strip_legacy_original_formats_line(q)
        original_formats.extend(from_legacy)

    # 3) Remove any "OriginalFormats:" line from the semantic query
    base_query, _ = strip_legacy_original_formats_line(q)

    # 3b) Remove repository/tool-name drift terms introduced by LLM tool calls.
    base_query = sanitize_retrieval_query(base_query, known_tool_names=get_known_names())

    # 4) Build soft format tokens (they bias but do not dominate)
    base_query = append_format_tokens(base_query, original_formats)

    # 5) Call retrieve() that includes automatic reranking
    hits = pipe.retrieve(
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
            candidates.append(CandidateDoc.model_validate(d.model_dump(mode="python")))
        except Exception:
            continue

    return SearchToolsOutput(candidates=candidates)
