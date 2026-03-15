from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel
from pydantic_ai.mcp import MCPServerStreamableHTTP

from ai_agent.agent.tools.utils import _clip
from ai_agent.agent.utils import _coerce_owner_repo_ref

log = logging.getLogger("agent.deepwiki")

# DeepWiki MCP server endpoint (Streamable HTTP transport)
DEEPWIKI_HTTP_URL = "https://mcp.deepwiki.com/mcp"

# Timeout for DeepWiki operations (seconds)
DEEPWIKI_TIMEOUT = 60


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
    Fetch repo docs from DeepWiki MCP (Streamable HTTP) and return a clipped string
    to keep LLM token usage under control.
    """
    owner, repo, _ = _coerce_owner_repo_ref(input.url)
    repo = f"{owner}/{repo}"

    try:
        server = MCPServerStreamableHTTP(DEEPWIKI_HTTP_URL)

        async with server:
            result = await asyncio.wait_for(
                server.direct_call_tool("read_wiki_contents", {"repoName": repo}),
                timeout=DEEPWIKI_TIMEOUT,
            )

            # Handle different result types from MCP
            text = None
            if isinstance(result, str):
                # Direct string result
                text = result
            elif hasattr(result, "content"):
                # MCP ToolResult with content field
                text_parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        text_parts.append(item.text)
                    elif isinstance(item, str):
                        text_parts.append(item)
                text = "\n".join(text_parts) if text_parts else None
            elif isinstance(result, list):
                # List of strings or content items
                text = "\n".join([str(p) for p in result if p]) or None

            if text and text.strip():
                clipped_text, truncated = _clip(text.strip())
                return DeepWikiContentsOutput(
                    success=True,
                    contents=clipped_text,
                    truncated=truncated,
                )

            return DeepWikiContentsOutput(
                success=False, error="No content returned from DeepWiki"
            )

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
