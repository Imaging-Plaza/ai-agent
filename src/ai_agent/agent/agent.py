from __future__ import annotations

import os, logging
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from generator.prompts import SELECTOR_SYSTEM
from generator.schema import CandidateDoc, Conversation, ConversationStatus, ToolSelection
from .models import AgentToolSelection, ToolRunLog
from .tools import (
    tool_search_tools, SearchToolsInput,
    tool_run_example, RunExampleInput,
    tool_fetch_url, FetchUrlInput,
)

log = logging.getLogger("agent.core")


class AgentState(BaseModel):
    """Placeholder for future conversational state (e.g., accumulated metadata)."""
    pass

# Build model wrapper (reuses OPENAI_MODEL / OPENAI_VLM_MODEL env if present)
MODEL_NAME = (
    os.getenv("OPENAI_VLM_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "gpt-4o-mini"
)

openai_model = OpenAIModel(MODEL_NAME)

# Agent definition -------------------------------------------------------------

agent = Agent(
    model=openai_model,
    system_prompt=(
        SELECTOR_SYSTEM
        + "\n\nYou now have TOOLS you may call before final answer."
        + "\nTools: search_tools(query, excluded, top_k) -> candidates; fetch_url(url) -> content; run_example(tool_name, image_path?) -> run log."\
        + "\nCall search_tools FIRST unless candidates already known. Use at most 3 tool calls then finalize."\
        + "\nWhen finalizing return ONLY valid JSON for ToolSelection schema."
    ),
    deps_type=AgentState,
)

# Register tools ---------------------------------------------------------------

@agent.tool()
async def search_tools(ctx: RunContext[AgentState], query: str, excluded: List[str] | None = None, top_k: int = 12):
    from .tools import tool_search_tools, SearchToolsInput
    out = tool_search_tools(SearchToolsInput(query=query, excluded=excluded or [], top_k=top_k))
    return [c.model_dump(mode="python") for c in out.candidates]

@agent.tool()
async def fetch_url(ctx: RunContext[AgentState], url: str):
    from .tools import tool_fetch_url, FetchUrlInput
    out = tool_fetch_url(FetchUrlInput(url=url))
    return out.model_dump(mode="python")

@agent.tool()
async def run_example(ctx: RunContext[AgentState], tool_name: str, image_path: str | None = None):
    from .tools import tool_run_example, RunExampleInput
    out = tool_run_example(RunExampleInput(tool_name=tool_name, image_path=image_path))
    return out.model_dump(mode="python")

# Runner wrapper ---------------------------------------------------------------

def run_agent(task: str, image_data_url: str | None = None, excluded: List[str] | None = None) -> AgentToolSelection:
    """Execute the agent. We inline the image as extra context in user message (multimodal reasoning)."""
    extra_context = ""
    if image_data_url:
        extra_context = "\nImage attached (PNG data URL). Consider modality/format cues."  # The actual binary is in upstream UI call to VLM; here we rely on metadata only for now.

    tool_logs: List[ToolRunLog] = []

    # Intercept tool usage by patching agent? Simpler: rely on return types (pydantic-ai tracks internally, we record manually not available yet) -> for Phase 1 we skip deep logging.

    result = agent.run_sync(task + extra_context, deps=AgentState(), output_type=ToolSelection).output

    # The agent's final content should be JSON for ToolSelection; we attempt parse via existing model.
    # import json
    # try:
    #     log.info("Agent output: %s", result.output)
    #     data = json.loads(result.output)
    # except Exception as e:
    #     # Fallback: ask for clarification
    #     conv = Conversation(status=ConversationStatus.NEEDS_CLARIFICATION, question="Clarify task (internal parse error)", context=str(e))
    #     return AgentToolSelection(conversation=conv, choices=[], tool_calls=tool_logs)

    # try:
    #     log.info("Agent output: %s", result.output)
    #     result = ToolSelection(**data)
    # except Exception as e:
    #     conv = Conversation(status=ConversationStatus.NEEDS_CLARIFICATION, question="Clarify task (schema mismatch)", context=str(e))
    #     return AgentToolSelection(conversation=conv, choices=[], tool_calls=tool_logs)

    return AgentToolSelection(
        conversation=result.conversation,
        choices=result.choices,
        explanation=result.explanation,
        reason=result.reason,
        tool_calls=tool_logs,
    )

__all__ = ["run_agent", "agent"]
