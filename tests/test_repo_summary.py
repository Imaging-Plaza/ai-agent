# tests/test_repo_summary.py
"""
Minimal smoke test for the repo summarizer tool.

Usage (from your project root):
    python tests/test_repo_summary.py \
        --url https://github.com/qchapp/lungs-segmentation \
        --out ./_out/lungs-segmentation-summary.md \
        --max-lines 12000 \
        --assert-contains segmentation CT

Notes:
- The tool uses the GitHub REST API. Set GITHUB_TOKEN in your environment to
  increase rate limits or access private repos.
- This script prints a short report to stdout and optionally writes the full
  Markdown summary to --out.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List
import sys
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src" / "ai_agent"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Import your tool
from ai_agent.agent.tools.repo_info_tool import (
    tool_repo_summary,
    RepoSummaryInput,
)


async def _run_summary(url: str):
    return await tool_repo_summary(RepoSummaryInput(url=url))


def run_summary(url: str):
    """
    Synchronous wrapper used by the smoke test to run the repo summary tool.
    """
    return asyncio.run(_run_summary(url))


DEFAULT_URL = "https://github.com/qchapp/lungs-segmentation"
DEFAULT_KEYWORDS: List[str] = ["segmentation", "CT"]


def test_repo_summary_smoke():
    """
    Minimal smoke test for the repo summarizer tool.

    This runs the summarizer against a known public repository and checks
    that a few expected keywords appear in the summary. It is intentionally
    lightweight and avoids making assertions about the full output structure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger("test.repo_summary")

    log.info("Summarizing repo: %s", DEFAULT_URL)
    res = run_summary(DEFAULT_URL)

    lower_summary = res.summary.lower()
    missing: List[str] = []
    for kw in DEFAULT_KEYWORDS:
        if kw.lower() not in lower_summary:
            missing.append(kw)

    if missing:
        log.error("Keywords not found in summary: %s", missing)

    assert not missing, f"Keywords not found in summary: {missing}"

    log.info(
        "Done. Summary length: %d chars, %d lines.",
        len(res.summary),
        len(res.summary.splitlines()),
    )
