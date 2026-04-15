from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

from ai_agent.generator.schema import CandidateDoc
from .utils import get_known_names, get_pipeline
from .query_utils import append_format_tokens, normalize_formats, sanitize_retrieval_query


class SearchAlternativeInput(BaseModel):
    """
    Input for searching with an alternative query formulation.

    Use this when initial search results are insufficient and you want to
    try a different phrasing or broader/narrower terms.
    """

    alternative_query: str = Field(
        description="Alternative query phrasing to try (can be similar terms, broader/narrower, etc.)"
    )
    excluded: List[str] = Field(default_factory=list)
    top_k: int = 12
    original_formats: List[str] = Field(default_factory=list)
    image_paths: List[str] = Field(default_factory=list)


class SearchAlternativeOutput(BaseModel):
    candidates: List[CandidateDoc]
    query_used: str


def tool_search_alternative(inp: SearchAlternativeInput) -> SearchAlternativeOutput:
    """
    Search with an alternative query formulation, with automatic reranking.

    This tool allows the agent to explicitly try a different search approach
    when initial results are not satisfactory.
    """
    pipe = get_pipeline()

    # Use the alternative query directly
    query = sanitize_retrieval_query(
        inp.alternative_query.strip(), known_tool_names=get_known_names()
    )

    # Normalize formats
    original_formats: List[str] = normalize_formats(inp.original_formats)
    query = append_format_tokens(query, original_formats)

    # Call retrieve() which includes automatic reranking
    hits = pipe.retrieve(
        query,
        image_paths=inp.image_paths or None,
        exclusions=inp.excluded,
        top_k=inp.top_k,
    )

    # Convert hits to CandidateDoc objects
    candidates: List[CandidateDoc] = []
    for h in hits:
        d = h.get("doc")
        if not d:
            continue
        try:
            candidates.append(CandidateDoc.model_validate(d.model_dump(mode="python")))
        except Exception:
            continue

    return SearchAlternativeOutput(
        candidates=candidates,
        query_used=query,
    )
