from __future__ import annotations

import re
import functools
from datetime import datetime
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic import BaseModel, Field
from typing import List, Optional, Set, Dict, Tuple, Any
from urllib.parse import urlparse


# Agent state to track tool usage ------------------------------------------------

class AgentState(BaseModel):
    """Holds incremental tool call logs for final reporting."""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_counts: Dict[str, int] = Field(default_factory=dict)
    disabled_tools: Set[str] = Field(default_factory=set)
    excluded_tools: List[str] = Field(default_factory=list)

    # Runtime overrides (session-only, not persisted)
    override_model: Optional[str] = None
    override_base_url: Optional[str] = None
    override_top_k: Optional[int] = None
    override_num_choices: Optional[int] = None

    image_paths: List[str] = Field(default_factory=list)
    original_formats: List[str] = Field(default_factory=list)

# Quota decorator + prepare hook -----------------------------------------------

QUOTA_PREFIX = "[TOOL_QUOTA_REACHED]"

class NonRetryableToolError(Exception):
    """Raised to indicate a quota/intentional stop that should not be retried."""
    pass

def limit_tool_calls(tool_name: str, cap: int, count_on_success: bool = True):
    """
    Enforce a per-run call limit for the given tool.

    - cap: hard maximum number of times this tool may be called in a single run.
    - count_on_success=True -> increments only if the call returns successfully.
      Set to False to count attempts even on failure.
    """
    def _decorate(fn):
        @functools.wraps(fn)
        async def _wrapped(*args, **kwargs):
            if not args or not isinstance(args[0], RunContext):
                # Defensive check: tools must receive ctx first
                raise NonRetryableToolError("Invalid tool signature: first argument must be RunContext.")

            ctx: RunContext[AgentState] = args[0]
            name = tool_name

            # If previously disabled, stop early with a non-retryable message.
            if name in ctx.deps.disabled_tools:
                raise NonRetryableToolError(f"{QUOTA_PREFIX} {name} is disabled for this run. Do not call it again.")

            current = ctx.deps.tool_counts.get(name, 0)
            if current >= cap:
                # Disable for the remainder of the run and log a synthetic blocked entry.
                ctx.deps.disabled_tools.add(name)
                ctx.deps.tool_calls.append({
                    "tool": name, "blocked": True, "reason": "quota", "cap": cap, "count": current, "timestamp": datetime.now().isoformat()
                })
                raise NonRetryableToolError(
                    f"{QUOTA_PREFIX} {name} usage limit reached (cap={cap}). "
                    f"Do not call this tool again; consider alternatives."
                )

            # Either count on success, or count attempts up-front.
            if count_on_success:
                result = await fn(*args, **kwargs)
                ctx.deps.tool_counts[name] = current + 1
                return result
            else:
                ctx.deps.tool_counts[name] = current + 1
                return await fn(*args, **kwargs)
        return _wrapped
    return _decorate


async def cap_prepare(ctx: RunContext["AgentState"], tool_def: ToolDefinition) -> ToolDefinition | None:
    """
    Prepare hook that hides tools after their quota is reached (or otherwise disabled).
    """
    if tool_def.name in ctx.deps.disabled_tools:
        return None
    return tool_def

_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

def _coerce_owner_repo_ref(input_str: str) -> Tuple[str, str, Optional[str]]:
    """
    Accepts:
      - https://github.com/owner/repo[.git][/tree/<ref>|#<ref>...]
      - http(s)://www.github.com/owner/repo...
      - github.com/owner/repo...
      - owner/repo
    Returns (owner, repo, ref|None) or raises ValueError with a helpful message.
    """
    s = (input_str or "").strip()

    # Short form: owner/repo
    if _OWNER_REPO_RE.match(s):
        owner, repo = s.split("/", 1)
        repo = repo.removesuffix(".git")
        return owner, repo, None

    # Missing scheme but github domain
    if s.startswith("github.com/") or s.startswith("www.github.com/"):
        s = "https://" + s.removeprefix("www.")

    # Full URL?
    try:
        u = urlparse(s)
    except Exception:
        u = None

    if u and u.netloc.lower() in {"github.com", "www.github.com"}:
        parts = [p for p in u.path.strip("/").split("/") if p]
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1].removesuffix(".git")
            ref = None
            # /tree/<ref> pattern
            if len(parts) >= 4 and parts[2] == "tree":
                ref = "/".join(parts[3:])
            # fragment as ref (e.g., #main)
            if u.fragment:
                ref = u.fragment
            return owner, repo, ref

    # As a last chance, try to extract owner/repo from a github-looking string
    m = re.search(r"github\.com[:/]+([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", s, re.I)
    if m:
        owner, repo = m.group(1), m.group(2).removesuffix(".git")
        return owner, repo, None

    raise ValueError(
        "[BAD_REPO_URL] Provide a GitHub repo as 'owner/repo' or a GitHub URL, "
        f"got: {input_str!r}"
    )

def coerce_github_url_or_none(s: str) -> str | None:
    """
    Returns a canonical GitHub URL ('https://github.com/owner/repo' or with #ref)
    if input is a valid GitHub repo URL or 'owner/repo'. Otherwise returns None.
    """
    try:
        owner, repo, ref = _coerce_owner_repo_ref(s)
        base = f"https://github.com/{owner}/{repo}"
        return f"{base}#{ref}" if ref else base
    except Exception:
        return None