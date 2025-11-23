from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel
from pydantic_ai.mcp import MCPServerSSE

from .utils import _clip
from ..utils import _coerce_owner_repo_ref

log = logging.getLogger("agent.deepwiki")

# DeepWiki MCP server endpoint (SSE transport)
DEEPWIKI_SSE_URL = "https://mcp.deepwiki.com/sse"

# Timeout for DeepWiki operations (seconds)
DEEPWIKI_TIMEOUT = 60

MAX_CHARS = 20000


class DeepWikiInput(BaseModel):
    """Input for DeepWiki operations."""
    url: str  # GitHub repository URL or owner/repo format


class DeepWikiContentsOutput(BaseModel):
    """Output from read_wiki_contents."""
    success: bool
    contents: Optional[str] = None
    error: Optional[str] = None
    truncated: bool = False


async def get_wiki_contents(input: DeepWikiInput) -> DeepWikiContentsOutput:
    """
    Fetch repo docs from DeepWiki MCP (SSE) and return a clipped string
    to keep LLM token usage under control.
    """
    owner, repo, _ = _coerce_owner_repo_ref(input.url)
    repo = f"{owner}/{repo}"

    try:
        server = MCPServerSSE(DEEPWIKI_SSE_URL)

        async with server:
            result = await asyncio.wait_for(
                server.direct_call_tool("read_wiki_contents", {"repoName": repo}),
                timeout=DEEPWIKI_TIMEOUT,
            )

            text = None
            if isinstance(result, list):
                text = "\n".join([p for p in result if isinstance(p, str)]) or None

            if text and text.strip():
                clipped_text, truncated = _clip(text.strip())
                return DeepWikiContentsOutput(
                    success=True,
                    contents=clipped_text,
                    truncated=truncated,
                )

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