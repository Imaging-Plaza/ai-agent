from __future__ import annotations

import os
from typing import Optional
import repocards

from pydantic import BaseModel

from .deepwiki_tool import get_wiki_contents, DeepWikiInput
from .utils import _clip, get_catalog_docs, _is_github_url

import logging

log = logging.getLogger("agent.repo_info")

# ----------------------------- Public I/O models ------------------------------


class RepoSummaryInput(BaseModel):
    url: str
    tool_name: Optional[str] = None


class RepoSummaryOutput(BaseModel):
    truncated: bool
    ref: Optional[str] = None
    summary: str
    source: str = "deepwiki"  # "deepwiki" or "repocards"


# ----------------------------- Tool entry point -------------------------------


async def tool_repo_summary(input: RepoSummaryInput) -> RepoSummaryOutput:
    """
    Summarize a GitHub repository using DeepWiki MCP (if available) or repocards fallback.

    Try DeepWiki first for fast, indexed docs. Falls back to repocards on any error.
    If URL is not a valid GitHub URL and tool_name is provided, attempts to retrieve
    the GitHub URL from the catalog.
    """
    effective_url = input.url

    # If URL doesn't look like a GitHub URL and tool_name is provided, try catalog lookup
    if input.tool_name and not _is_github_url(input.url):
        log.info(
            f"Non-GitHub URL provided ({input.url}), attempting catalog lookup for tool: {input.tool_name}"
        )
        try:
            catalog_url = _get_repo_url_from_catalog(input.tool_name)
            if catalog_url:
                log.info(
                    f"Found GitHub URL in catalog for {input.tool_name}: {catalog_url}"
                )
                effective_url = catalog_url
            else:
                log.warning(f"No GitHub URL found in catalog for {input.tool_name}")
        except Exception as e:
            log.warning(f"Failed to lookup repo URL from catalog: {e}")

    # Try DeepWiki first
    try:
        log.info(f"Attempting DeepWiki lookup for {effective_url}")
        deepwiki_result = await get_wiki_contents(DeepWikiInput(url=effective_url))

        if deepwiki_result.success and deepwiki_result.contents:
            log.info("DeepWiki lookup succeeded")
            summary = f"# Repository Documentation (via DeepWiki)\n\n{deepwiki_result.contents}"
            return RepoSummaryOutput(
                truncated=deepwiki_result.truncated,
                ref=None,
                summary=summary,
                source="deepwiki",
            )
        else:
            log.warning(f"DeepWiki lookup failed: {deepwiki_result.error}")
    except Exception as e:
        log.warning(f"DeepWiki error (falling back to repocards): {e}")

    # Fallback to repocards
    log.info(f"Using repocards fallback for {effective_url}")
    try:
        summary, truncated = _clip(
            repocards.get_repo_info(
                effective_url, github_token=os.getenv("GITHUB_TOKEN")
            )
        )
        return RepoSummaryOutput(
            truncated=truncated, ref=None, summary=summary, source="repocards"
        )
    except Exception as e:
        log.error(f"Repocards fallback also failed: {e}")
        return RepoSummaryOutput(
            truncated=False,
            ref=None,
            summary=f"# Error\n\nFailed to fetch repository information: {str(e)}",
            source="error",
        )


def _get_repo_url_from_catalog(tool_name: str) -> Optional[str]:
    """
    Look up a tool's GitHub repository URL from the catalog.

    Uses lightweight catalog reading to avoid initializing the full RAG pipeline
    (embedder/reranker/index) for simple metadata lookups.

    Args:
        tool_name: Name of the tool to look up

    Returns:
        GitHub URL if found, None otherwise
    """
    try:
        # Use lightweight catalog reader instead of full pipeline
        docs = get_catalog_docs()

        # Look up tool by name (case-insensitive)
        tool_name_lower = tool_name.lower().strip()
        for doc in docs:
            if doc.name.lower().strip() == tool_name_lower:
                # Try repo_url field first, then url field
                if doc.repo_url and _is_github_url(doc.repo_url):
                    return doc.repo_url

                if doc.url and _is_github_url(doc.url):
                    return doc.url

        return None
    except Exception as e:
        log.error(f"Error looking up repo URL from catalog: {e}")
        return None
