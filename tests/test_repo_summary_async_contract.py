from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ai_agent.agent.tools.repo_info_tool import RepoSummaryInput, tool_repo_summary
from ai_agent.agent.tools.deepwiki_tool import DeepWikiContentsOutput


def test_tool_repo_summary_is_async():
    assert inspect.iscoroutinefunction(tool_repo_summary)


@pytest.mark.asyncio
async def test_tool_repo_summary_async_invocation():
    with patch(
        "ai_agent.agent.tools.repo_info_tool.get_wiki_contents",
        new_callable=AsyncMock,
    ) as mock_deepwiki:
        mock_deepwiki.return_value = DeepWikiContentsOutput(
            success=True,
            contents="# Repo\n\nSummary",
            truncated=False,
        )

        out = await tool_repo_summary(RepoSummaryInput(url="owner/repo"))
        assert out.source == "deepwiki"
        assert "Repo" in out.summary
