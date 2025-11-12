from __future__ import annotations

import os
import asyncio
import logging
from typing import Optional

from pydantic import BaseModel
from pydantic_ai.mcp import MCPServerSSE

log = logging.getLogger("agent.deepwiki")

# DeepWiki MCP server endpoint (SSE transport)
DEEPWIKI_SSE_URL = "https://mcp.deepwiki.com/sse"

# Timeout for DeepWiki operations (seconds)
DEEPWIKI_TIMEOUT = 30

MAX_CHARS = 20000


class DeepWikiInput(BaseModel):
    """Input for DeepWiki operations."""
    url: str  # GitHub repository URL or owner/repo format


class DeepWikiContentsOutput(BaseModel):
    """Output from read_wiki_contents."""
    success: bool
    contents: Optional[str] = None
    error: Optional[str] = None


def _normalize_github_url(url: str) -> str:
    """
    Normalize GitHub URL to owner/repo format expected by DeepWiki.
    
    Examples:
        - https://github.com/owner/repo -> owner/repo
        - owner/repo -> owner/repo
    """
    url = url.strip()
    
    # Remove protocol and www
    for prefix in ["https://", "http://", "www."]:
        if url.startswith(prefix):
            url = url[len(prefix):]
    
    # Remove github.com/ prefix
    if url.startswith("github.com/"):
        url = url[len("github.com/"):]
    
    # Remove .git suffix
    if url.endswith(".git"):
        url = url[:-4]
    
    # Remove trailing slashes and any path components beyond owner/repo
    parts = [p for p in url.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    
    return url


async def get_wiki_contents(input: DeepWikiInput) -> DeepWikiContentsOutput:
    """
    Fetch repo docs from DeepWiki MCP (SSE) and return a clipped string
    to keep LLM token usage under control.
    """
    repo = _normalize_github_url(input.url)

    def _clip(s: str) -> str:
        if not s:
            return s
        if len(s) <= MAX_CHARS:
            return s
        return s[:MAX_CHARS] + "\n\n...[truncated for token budget]..."

    try:
        server = MCPServerSSE(DEEPWIKI_SSE_URL)

        async with server:
            # Sanity: tool exists
            tools = await server.list_tools()
            if not any(getattr(t, "name", "") == "read_wiki_contents" for t in tools):
                return DeepWikiContentsOutput(
                    success=False,
                    error="read_wiki_contents tool not found on DeepWiki server",
                )

            result = await asyncio.wait_for(
                server.direct_call_tool("read_wiki_contents", {"repoName": repo}),
                timeout=DEEPWIKI_TIMEOUT,
            )

            # Normalize result into a text blob
            text = None
            if isinstance(result, str):
                text = result
            elif isinstance(result, list):
                text = "\n".join([p for p in result if isinstance(p, str)]) or None
            elif isinstance(result, dict):
                for key in ("text", "content", "markdown"):
                    v = result.get(key)
                    if isinstance(v, str):
                        text = v
                        break
                    if isinstance(v, list):
                        parts = []
                        for item in v:
                            if isinstance(item, str):
                                parts.append(item)
                            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                                parts.append(item["text"])
                        if parts:
                            text = "\n".join(parts)
                            break
            if text is None:
                # Fallback for SDK-like objects
                content_attr = getattr(result, "content", None)
                if content_attr:
                    parts = []
                    for item in content_attr:
                        t = getattr(item, "text", None)
                        if isinstance(t, str):
                            parts.append(t)
                    if parts:
                        text = "\n".join(parts)

            if text and text.strip():
                print(text.strip()[:500])  # Log first 500 chars
                return DeepWikiContentsOutput(success=True, contents=_clip(text.strip()))

            return DeepWikiContentsOutput(success=False, error="No content returned from DeepWiki")

    except asyncio.TimeoutError:
        log.warning(f"DeepWiki timed out after {DEEPWIKI_TIMEOUT}s for {repo}")
        return DeepWikiContentsOutput(
            success=False,
            error=f"DeepWiki request timed out after {DEEPWIKI_TIMEOUT}s",
        )
    except Exception as e:
        log.error(f"Failed to get wiki contents for {repo}: {e}")
        return DeepWikiContentsOutput(
            success=False,
            error=f"Failed to connect to DeepWiki: {str(e)}",
        )


__all__ = [
    "get_wiki_contents",
    "DeepWikiInput",
    "DeepWikiContentsOutput",
]
