from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

from ai_agent.generator.schema import CandidateDoc
from .utils import get_pipeline


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
    Search with an alternative query formulation.
    
    This tool allows the agent to explicitly try a different search approach
    when initial results are not satisfactory.
    """
    pipe = get_pipeline()
    
    # Use the alternative query directly
    query = inp.alternative_query.strip()
    
    # Normalize formats
    original_formats: List[str] = [f.lower() for f in inp.original_formats]
    
    # Build soft format tokens
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
        query = (
            query + " " + " ".join(f"format:{t}" for t in fmt_tokens)
        ).strip()
    
    # Call retrieval with the alternative query
    # Set min_results=0 to prevent automatic retry (agent is already retrying)
    hits = pipe.retrieve_no_rerank(
        query,
        image_paths=inp.image_paths or None,
        exclusions=inp.excluded,
        top_k=inp.top_k,
        min_results=0,  # Disable automatic retry since agent controls this
        max_retries=0,  # Disable automatic retry
    )
    
    # Convert hits to CandidateDoc objects
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
    
    return SearchAlternativeOutput(
        candidates=candidates,
        query_used=query,
    )
