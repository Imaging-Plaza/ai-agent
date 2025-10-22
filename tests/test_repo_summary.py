# scripts/test_repo_summary.py
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

import argparse
import logging
import os
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
from agent.tools.repo_info_tool import (
    tool_repo_summary,
    RepoSummaryInput,
)

def main():
    parser = argparse.ArgumentParser(description="Smoke test for repo summary tool")
    parser.add_argument(
        "--url",
        default="https://github.com/qchapp/lungs-segmentation",
        help="GitHub repo URL to summarize",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the full Markdown summary",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=100,
        help="Max lines to print to stdout (for brevity). Use a large number for full output.",
    )
    parser.add_argument(
        "--assert-contains",
        nargs="*",
        default=[],
        help="Optional list of keywords the summary must contain (case-insensitive).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger("test.repo_summary")

    # Run the tool
    log.info("Summarizing repo: %s", args.url)
    res = tool_repo_summary(RepoSummaryInput(url=args.url))

    # Basic report
    print("\n=== Repo Summary (header) ===")
    print(f"Ref: {res.ref} | Truncated: {res.truncated}")
    print("=============================\n")

    # Print a truncated view to stdout for quick inspection
    lines = res.summary.splitlines()
    head = lines[: args.max_lines]
    print("\n".join(head))
    if len(lines) > args.max_lines:
        print("\n... (truncated for stdout; full summary may be longer)")

    # Optional: write full Markdown summary
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(res.summary, encoding="utf-8")
        log.info("Wrote full summary to: %s", out_path)

    # Optional assertions
    failed: List[str] = []
    if args.assert_contains:
        lower_summary = res.summary.lower()
        for kw in args.assert_contains:
            if kw.lower() not in lower_summary:
                failed.append(kw)
        if failed:
            msg = f"Assertion failed: keywords not found in summary: {failed}"
            log.error(msg)
            raise SystemExit(2)

    log.info("Done. Summary length: %d chars, %d lines.", len(res.summary), len(lines))


if __name__ == "__main__":
    # Allow running with `python scripts/test_repo_summary.py`
    main()
