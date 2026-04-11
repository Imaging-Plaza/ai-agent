from __future__ import annotations

import asyncio
import os
import time
from typing import Optional
import repocards

from pydantic import BaseModel

from .deepwiki_tool import get_wiki_contents, DeepWikiInput
from .utils import _clip, get_catalog_docs, _is_github_url

import logging

log = logging.getLogger("agent.repo_info")

REPO_INFO_CACHE_TTL_SECONDS = int(os.getenv("REPO_INFO_CACHE_TTL_SECONDS", "3600"))
REPO_INFO_CACHE_MAX_ENTRIES = int(os.getenv("REPO_INFO_CACHE_MAX_ENTRIES", "256"))

_REPO_INFO_CACHE: dict[str, tuple[float, "RepoSummaryOutput"]] = {}
_REPO_INFO_INFLIGHT: dict[str, asyncio.Future["RepoSummaryOutput"]] = {}
_REPO_INFO_LOCK = asyncio.Lock()

# ----------------------------- Public I/O models ------------------------------


class RepoSummaryInput(BaseModel):
    url: str
    tool_name: Optional[str] = None


class RepoSummaryOutput(BaseModel):
    truncated: bool
    ref: Optional[str] = None
    summary: str
    source: str = "deepwiki"  # "deepwiki" or "repocards"


def _normalize_cache_key(url: str) -> str:
    return url.strip().lower()


def _evict_expired_entries(now: float) -> None:
    expired = [k for k, (exp, _) in _REPO_INFO_CACHE.items() if exp <= now]
    for key in expired:
        _REPO_INFO_CACHE.pop(key, None)


def _enforce_cache_capacity() -> None:
    if len(_REPO_INFO_CACHE) <= REPO_INFO_CACHE_MAX_ENTRIES:
        return
    # Evict oldest expiration first. Good enough for bounded in-memory cache.
    keys_by_expiry = sorted(_REPO_INFO_CACHE.items(), key=lambda item: item[1][0])
    over = len(_REPO_INFO_CACHE) - REPO_INFO_CACHE_MAX_ENTRIES
    for key, _ in keys_by_expiry[:over]:
        _REPO_INFO_CACHE.pop(key, None)


def _clear_repo_summary_cache_for_tests() -> None:
    """Test helper to avoid state leakage across test cases."""
    _REPO_INFO_CACHE.clear()
    _REPO_INFO_INFLIGHT.clear()


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

    cache_key = _normalize_cache_key(effective_url)
    now = time.monotonic()

    await _REPO_INFO_LOCK.acquire()
    try:
        _evict_expired_entries(now)

        cached_entry = _REPO_INFO_CACHE.get(cache_key)
        if cached_entry:
            _, cached = cached_entry
            log.info(f"Repo info cache hit for {effective_url}")
            return cached.model_copy(deep=True)

        inflight = _REPO_INFO_INFLIGHT.get(cache_key)
        if inflight is None:
            inflight = asyncio.get_running_loop().create_future()
            _REPO_INFO_INFLIGHT[cache_key] = inflight
            creator = True
        else:
            creator = False
    finally:
        _REPO_INFO_LOCK.release()

    if not creator:
        log.info(f"Repo info in-flight dedup for {effective_url}")
        shared = await inflight
        return shared.model_copy(deep=True)

    try:
        result = await _fetch_repo_summary(effective_url)
    except BaseException as exc:
        await _REPO_INFO_LOCK.acquire()
        try:
            if not inflight.done():
                inflight.set_exception(exc)
            _REPO_INFO_INFLIGHT.pop(cache_key, None)
        finally:
            _REPO_INFO_LOCK.release()
        raise

    await _REPO_INFO_LOCK.acquire()
    try:
        if result.source != "error" and REPO_INFO_CACHE_TTL_SECONDS > 0:
            expires_at = time.monotonic() + REPO_INFO_CACHE_TTL_SECONDS
            _REPO_INFO_CACHE[cache_key] = (expires_at, result)
            _enforce_cache_capacity()

        if not inflight.done():
            inflight.set_result(result)
        _REPO_INFO_INFLIGHT.pop(cache_key, None)
    finally:
        _REPO_INFO_LOCK.release()

    return result.model_copy(deep=True)


async def _fetch_repo_summary(effective_url: str) -> RepoSummaryOutput:
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
