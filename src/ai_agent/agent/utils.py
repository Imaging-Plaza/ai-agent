from __future__ import annotations

import functools
from datetime import datetime
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Set, Dict


# Agent state to track tool usage ------------------------------------------------

class AgentState(BaseModel):
    """Holds incremental tool call logs for final reporting."""
    tool_calls: List[dict] = []  # (kept as-is to not modify existing working field)
    tool_counts: Dict[str, int] = Field(default_factory=dict)
    disabled_tools: Set[str] = Field(default_factory=set)
    excluded_tools: List[str] = Field(default_factory=list)  # Tools to exclude from search

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