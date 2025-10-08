from __future__ import annotations

import os, logging
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from generator.prompts import AGENT_SYSTEM_PROMPT
from generator.schema import ToolSelection
from api.pipeline import RAGImagingPipeline
from utils.utils import _best_runnable_link
from .models import AgentToolSelection, ToolRunLog
from .tools import (
    tool_search_tools, SearchToolsInput,
    tool_rerank, RerankInput,
    # tool_run_example, RunExampleInput,
    tool_repo_info, RepoInfoInput,
)

log = logging.getLogger("agent.core")


class AgentState(BaseModel):
    """Holds incremental tool call logs for final reporting."""
    tool_calls: List[dict] = []

MODEL_NAME = (
    os.getenv("OPENAI_VLM_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "gpt-4o-mini"
)

openai_model = OpenAIModel(MODEL_NAME)

# Agent definition -------------------------------------------------------------

agent = Agent(
    model=openai_model,
    system_prompt=AGENT_SYSTEM_PROMPT,
    deps_type=AgentState,
)

# Register tools ---------------------------------------------------------------

@agent.tool()
async def search_tools(ctx: RunContext[AgentState], query: str, excluded: List[str] | None = None, top_k: int = 12, original_formats: List[str] | None = None):
    out = tool_search_tools(SearchToolsInput(query=query, excluded=excluded or [], top_k=top_k, original_formats=original_formats or []))
    payload = [c.model_dump(mode="python") for c in out.candidates]
    ctx.deps.tool_calls.append({"tool": "search_tools", "query": query, "count": len(payload), "original_formats": original_formats or []})
    return payload

@agent.tool()
async def rerank(ctx: RunContext[AgentState], query: str, candidate_names: List[str], top_k: int = 5):
    out = tool_rerank(RerankInput(query=query, candidate_names=candidate_names, top_k=top_k))
    ctx.deps.tool_calls.append({"tool": "rerank", "query": query, "used_model": out.used_model, "count": len(out.reranked)})
    return out.model_dump(mode="python")

# @agent.tool()
# async def run_example(ctx: RunContext[AgentState], tool_name: str, image_path: str | None = None):
#     out = tool_run_example(RunExampleInput(tool_name=tool_name, image_path=image_path))
#     ctx.deps.tool_calls.append({"tool": "run_example", "tool_name": tool_name, "ran": out.ran})
#     return out.model_dump(mode="python")

@agent.tool()
async def repo_info(ctx: RunContext[AgentState], url: str):
    out = tool_repo_info(RepoInfoInput(url=url))
    ctx.deps.tool_calls.append({"tool": "repo_info", "url": url, "truncated": out.truncated})
    return out.model_dump(mode="python")

@agent.tool()
async def resolve_demo_link(ctx: RunContext[AgentState], tool_name: str):
    """Return the best runnable demo link for a tool (if any)."""
    link = None
    try:
        pipe = RAGImagingPipeline(docs=[])
        doc = pipe.get_doc(tool_name)
        if doc:
            link = _best_runnable_link(doc)
    except Exception:
        link = None
    ctx.deps.tool_calls.append({"tool": "resolve_demo_link", "tool_name": tool_name, "demo_link": link})
    return {"tool_name": tool_name, "demo_link": link}

# Runner wrapper ---------------------------------------------------------------

def run_agent(task: str, image_data_url: str | None = None, excluded: List[str] | None = None,
              original_formats: List[str] | None = None, image_meta: str | None = None) -> AgentToolSelection:
    """Execute the agent. We inline the image as extra context in user message (multimodal reasoning)."""
    extra_context = ""
    if image_data_url:
        # Neutral preview line that avoids implying original format
        extra_context = "\nPreview image provided (rendered PNG). DO NOT infer original format from this preview; rely on 'OriginalFormats:' line if present."

    tool_logs: List[ToolRunLog] = []

    # Intercept tool usage by patching agent? Simpler: rely on return types (pydantic-ai tracks internally, we record manually not available yet) -> for Phase 1 we skip deep logging.

    deps = AgentState()
    # Provide hidden metadata context lines (non-user-visible) below a delimiter
    hidden_meta = ""
    if original_formats:
        hidden_meta += "\n(Formats Hint: " + ",".join(original_formats) + ")"
    if image_meta:
        # collapse newlines to avoid confusing the model with too many lines
        short_meta = " ".join(x.strip() for x in image_meta.splitlines() if x.strip())
        hidden_meta += "\n(Image Metadata: " + short_meta[:500] + ("…" if len(short_meta) > 500 else "") + ")"
    prompt = task + extra_context + hidden_meta
    log.info(f"Agent prompt: {prompt}")
    result = agent.run_sync(prompt, deps=deps, output_type=ToolSelection).output

    # Convert tool call dicts into ToolRunLog entries
    for tc in deps.tool_calls:
        tool_logs.append(ToolRunLog(tool=tc.get("tool"), inputs={k: v for k, v in tc.items() if k not in {"tool"}}, summary=str(tc)))

    # Post-run enrichment: pull demo links from resolve_demo_link tool calls
    demo_map = {tc.inputs.get("tool_name"): tc.inputs.get("demo_link") for tc in tool_logs if tc.tool == "resolve_demo_link" and tc.inputs.get("tool_name")}
    for ch in result.choices:
        if getattr(ch, 'name', None) and ch.name in demo_map and demo_map[ch.name]:
            setattr(ch, 'demo_link', demo_map[ch.name])

    return AgentToolSelection(
        conversation=result.conversation,
        choices=result.choices,
        explanation=result.explanation,
        reason=result.reason,
        tool_calls=tool_logs,
    )

__all__ = ["run_agent", "agent"]
