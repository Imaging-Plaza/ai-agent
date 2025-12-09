from __future__ import annotations

import os
from typing import Optional
import repocards

from pydantic import BaseModel

from .deepwiki_tool import get_wiki_contents, DeepWikiInput
from .utils import _clip

import logging
log = logging.getLogger("agent.repo_info")

# ----------------------------- Public I/O models ------------------------------

class RepoSummaryInput(BaseModel):
    url: str


class RepoSummaryOutput(BaseModel):
    truncated: bool
    ref: Optional[str] = None
    summary: str
    source: str = "deepwiki"  # "deepwiki" or "repocards"

# ----------------------------- Tool entry point -------------------------------

async def tool_repo_summary(input: RepoSummaryInput) -> RepoSummaryOutput:
    """
    Summarize a GitHub repository using DeepWiki MCP (if available) or GitHub API fallback.
    
    Try DeepWiki first for fast, indexed docs. Falls back to GitHub API on any error.
    """
    # Try DeepWiki first
    try:
        log.info(f"Attempting DeepWiki lookup for {input.url}")
        deepwiki_result = await get_wiki_contents(DeepWikiInput(url=input.url))
        
        if deepwiki_result.success and deepwiki_result.contents:
            log.info("DeepWiki lookup succeeded")
            summary = f"# Repository Documentation (via DeepWiki)\n\n{deepwiki_result.contents}"
            return RepoSummaryOutput(
                truncated=deepwiki_result.truncated,
                ref=None,
                summary=summary,
                source="deepwiki"
            )
        else:
            log.warning(f"DeepWiki lookup failed: {deepwiki_result.error}")
    except Exception as e:
        log.warning(f"DeepWiki error (falling back to repocards): {e}")
    
    # Fallback to repocards
    log.info(f"Using repocards fallback for {input.url}")
    summary, truncated = _clip(repocards.get_repo_info(input.url, github_token=os.getenv("GITHUB_TOKEN")))
    return RepoSummaryOutput(
        truncated=truncated,
        ref=None,
        summary=summary,
        source="repocards"
    )