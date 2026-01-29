from __future__ import annotations

import os, logging
from datetime import datetime
from typing import List

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ai_agent.generator.prompts import get_agent_system_prompt
from ai_agent.generator.schema import ToolSelection
from ai_agent.api.pipeline import RAGImagingPipeline
from ai_agent.utils.utils import _best_runnable_link
from ai_agent.utils.config import get_config
from .models import AgentToolSelection, ToolRunLog
from .tools.repo_info_tool import (
    tool_repo_summary,
    RepoSummaryInput,
    coerce_github_url_or_none,
)
from .tools.search_tool import tool_search_tools, SearchToolsInput
from .tools.search_alternative_tool import tool_search_alternative, SearchAlternativeInput
from .tools.gradio_space_tool import tool_run_example, RunExampleInput
from .utils import AgentState, limit_tool_calls, cap_prepare
from ai_agent.utils.image_meta import summarize_image_metadata, detect_ext_token

log = logging.getLogger("agent.core")

# ---------------------------------------------------------------------------
# Model / provider setup
# ---------------------------------------------------------------------------
config = get_config()
agent_model_config = config.agent_model

try:
    api_key = agent_model_config.get_api_key()
except ValueError as e:
    log.error(f"Failed to get API key for agent model: {e}")
    raise

log.info(f"Initializing agent model: {agent_model_config.name}")

if agent_model_config.base_url:
    log.info(f"Using custom OpenAI base URL: {agent_model_config.base_url}")
    provider = OpenAIProvider(
        base_url=agent_model_config.base_url,
        api_key=api_key,
    )
else:
    provider = OpenAIProvider(api_key=api_key)

openai_model = OpenAIChatModel(
    model_name=agent_model_config.name,
    provider=provider,
)

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
agent = Agent(
    model=openai_model,
    system_prompt=get_agent_system_prompt(os.getenv("NUM_CHOICES", "3")),
    deps_type=AgentState,
)

# ---------------------------------------------------------------------------
# Tool adapters for the agent
# ---------------------------------------------------------------------------

@agent.tool(retries=2, prepare=cap_prepare)
@limit_tool_calls("search_tools", cap=1)
async def search_tools(
    ctx: RunContext[AgentState],
    query: str,
    excluded: List[str] | None = None,
    top_k: int = 12,
) -> List[dict]:
    """
    Agent-facing search tool.

    Delegates to tools.search_tool.tool_search_tools(), but automatically
    injects:
      - globally excluded tools (from ctx.deps.excluded_tools)
      - image_paths and original_formats (from ctx.deps, set in run_agent)
    so the language model never has to reason about file paths directly.
    """
    # Merge explicit exclusions with global exclusions from AgentState
    explicit_excluded = excluded or []
    global_excluded = getattr(ctx.deps, "excluded_tools", []) or []
    all_excluded = sorted(set(explicit_excluded + list(global_excluded)))

    original_formats = getattr(ctx.deps, "original_formats", []) or []
    image_paths = getattr(ctx.deps, "image_paths", []) or []

    effective_top_k = ctx.deps.override_top_k if ctx.deps.override_top_k is not None else top_k

    inp = SearchToolsInput(
        query=query,
        excluded=all_excluded,
        top_k=effective_top_k,
        original_formats=original_formats,
        image_paths=image_paths,
    )
    out = tool_search_tools(inp)

    ctx.deps.tool_calls.append(
        {
            "tool": "search_tools",
            "query": query,
            "count": len(out.candidates),
            "original_formats": original_formats,
            "excluded": all_excluded,
            "timestamp": datetime.now().isoformat()
        }
    )

    # Return plain dicts so the LLM sees a simple JSON-like structure.
    return [c.model_dump(mode="python") for c in out.candidates]


@agent.tool(retries=2, prepare=cap_prepare)
@limit_tool_calls("search_alternative", cap=3)
async def search_alternative(
    ctx: RunContext[AgentState],
    alternative_query: str,
    excluded: List[str] | None = None,
    top_k: int = 12,
) -> List[dict]:
    """
    Search with an alternative query formulation (includes automatic reranking).
    """
    # Merge exclusions
    explicit_excluded = excluded or []
    global_excluded = getattr(ctx.deps, "excluded_tools", []) or []
    all_excluded = sorted(set(explicit_excluded + list(global_excluded)))

    original_formats = getattr(ctx.deps, "original_formats", []) or []
    image_paths = getattr(ctx.deps, "image_paths", []) or []

    inp = SearchAlternativeInput(
        alternative_query=alternative_query,
        excluded=all_excluded,
        top_k=top_k,
        original_formats=original_formats,
        image_paths=image_paths,
    )
    out = tool_search_alternative(inp)

    ctx.deps.tool_calls.append(
        {
            "tool": "search_alternative",
            "alternative_query": alternative_query,
            "query_used": out.query_used,
            "count": len(out.candidates),
            "original_formats": original_formats,
            "excluded": all_excluded,
            "timestamp": datetime.now().isoformat()
        }
    )

    return [c.model_dump(mode="python") for c in out.candidates]


@agent.tool(retries=2, prepare=cap_prepare)
@limit_tool_calls("repo_info", cap=12)
async def repo_info(ctx: RunContext[AgentState], url: str) -> dict:
    """
    Fetch a short summary of a GitHub repository.

    Non-GitHub URLs are ignored; the tool returns a small dict noting
    that it was skipped.
    """
    norm_url = coerce_github_url_or_none(url)
    if not norm_url:
        payload = {
            "tool": "repo_info",
            "url": url,
            "skipped": True,
            "reason": "NON_GITHUB_URL",
            "hint": "Pass a GitHub repo URL or 'owner/repo' to repo_info(url).",
            "timestamp": datetime.now().isoformat()
        }
        ctx.deps.tool_calls.append(payload)
        return {k: v for k, v in payload.items() if k != "tool"}

    try:
        out = tool_repo_summary(RepoSummaryInput(url=norm_url))
    except Exception as e:
        ctx.deps.tool_calls.append(
            {"tool": "repo_info", "url": norm_url, "error": str(e), "timestamp": datetime.now().isoformat()}
        )
        raise

    ctx.deps.tool_calls.append(
        {
            "tool": "repo_info",
            "url": norm_url,
            "truncated": getattr(out, "truncated", False),
            "timestamp": datetime.now().isoformat()
        }
    )
    return out.model_dump(mode="python")


@agent.tool(retries=0, prepare=cap_prepare)
@limit_tool_calls("run_example", cap=1)
async def run_example(
    ctx: RunContext[AgentState],
    tool_name: str,
    endpoint_url: str | None = None,
    extra_text: str | None = None,
) -> dict:
    """
    Run an example / demo for a given tool via its Gradio space.

    Thin wrapper around tools.gradio_space_tool.tool_run_example().
    """
    out = tool_run_example(
        RunExampleInput(
            tool_name=tool_name,
            endpoint_url=endpoint_url,
            extra_text=extra_text,
        )
    )
    ctx.deps.tool_calls.append(
        {
            "tool": "run_example",
            "tool_name": tool_name,
            "ran": getattr(out, "ran", False),
            "endpoint_url": getattr(out, "endpoint_url", endpoint_url),
            "api_name": getattr(out, "api_name", None),
            "timestamp": datetime.now().isoformat(),
        }
    )
    return out.model_dump(mode="python")


# ---------------------------------------------------------------------------
# High level entry point: run the agent on (text query + image)
# ---------------------------------------------------------------------------
def run_agent(
    task: str,
    image_paths: List[str],
    excluded: List[str] | None = None,
    conversation_history: List[str] | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    top_k: int | None = None,
    num_choices: int | None = None,
    image_metadata: str | None = None,
) -> AgentToolSelection:
    """
    Execute the agent for a user task and at least one image path.

    - derive canonical original_formats (tiff / dicom / nifti / ...)
    - build a compact image metadata summary (or use pre-computed one)
    - pass both to the LLM as hidden context
    - store image_paths/original_formats in deps so retrieval tools can use them
    - optionally allow runtime model/base_url/top_k/num_choices overrides
    
    Args:
        image_metadata: Optional pre-computed metadata string. If provided,
                       avoids redundant metadata extraction.
    """
    if not image_paths:
        raise ValueError("run_agent requires at least one image path")

    tool_logs: List[ToolRunLog] = []

    # ---- 1) Derive image-based metadata and format hints --------------------
    # Use pre-computed metadata if available, otherwise compute it
    meta_str = image_metadata if image_metadata is not None else (summarize_image_metadata(image_paths) or "")
    fmt_str = detect_ext_token(image_paths) or ""
    original_formats = [t.lower() for t in fmt_str.split()] if fmt_str else []

    # ---- 2) Prepare dependency state passed to all tools --------------------
    # Keep the "excluded_tools" pattern from develop, but also keep your overrides.
    deps = AgentState(
        excluded_tools=excluded or [],
        override_model=model,
        override_base_url=base_url,
        override_top_k=top_k,
        override_num_choices=num_choices,
    )

    # Store image information on deps so tools can reuse it.
    setattr(deps, "image_paths", list(image_paths))
    setattr(deps, "original_formats", original_formats)

    # ---- 3) Hidden metadata lines for the model ----------------------------
    hidden_meta = ""
    if original_formats:
        hidden_meta += "\n(Formats Hint: " + ",".join(original_formats) + ")"
    if meta_str:
        short_meta = " ".join(x.strip() for x in meta_str.splitlines() if x.strip())
        hidden_meta += "\n(Image Metadata: " + short_meta[:500] + ("…" if len(short_meta) > 500 else "") + ")"
    if top_k is not None:
        hidden_meta += f"\n(Search top_k: {top_k})"

    # Visible hint so the model remembers there *is* an image.
    extra_context = "\nPreview image provided. Use tools compatible with its modality, anatomy, and file format."

    # ---- 4) Build the prompt (optionally including history) ----------------
    if conversation_history and len(conversation_history) > 0:
        history_text = "\n".join(conversation_history)
        prompt = (
            f"Previous conversation:\n{history_text}\n\n"
            f"Current request: {task}{extra_context}{hidden_meta}"
        )
    else:
        prompt = task + extra_context + hidden_meta

    # -----------------------------------------------------------------------
    # Determine which agent instance to use (YOUR FEATURE — kept)
    # -----------------------------------------------------------------------
    agent_instance = agent  # Default to global agent
    effective_num_choices = num_choices if num_choices is not None else 3
    effective_model = model if model else agent_model_config.name
    effective_top_k = top_k if top_k is not None else 12

    # When model is provided from UI, base_url comes with it (can be None for OpenAI)
    # When model is NOT provided, use config defaults
    if model:
        # Model selected from dropdown - base_url parameter is authoritative
        if base_url and "inference.rcp.epfl.ch" in base_url:
            # EPFL model selected
            runtime_api_key = os.getenv("EPFL_API_KEY")
            if not runtime_api_key:
                raise ValueError(
                    "EPFL_API_KEY not found. Cannot use EPFL models without VPN and API key."
                )
            effective_base_url = base_url
            log.info("✓ Using EPFL_API_KEY for EPFL inference server")
        else:
            # OpenAI or other model selected (base_url=None means OpenAI)
            runtime_api_key = os.getenv("OPENAI_API_KEY")
            if not runtime_api_key:
                raise ValueError("OPENAI_API_KEY not found. Cannot use OpenAI models.")
            effective_base_url = base_url  # Will be None for OpenAI
            log.info("✓ Using OPENAI_API_KEY for OpenAI endpoint")
    else:
        # No model override - use config defaults
        effective_base_url = agent_model_config.base_url
        if effective_base_url and "inference.rcp.epfl.ch" in effective_base_url:
            runtime_api_key = os.getenv("EPFL_API_KEY")
            if not runtime_api_key:
                raise ValueError("EPFL_API_KEY not found")
            log.info("✓ Using EPFL_API_KEY from config")
        else:
            runtime_api_key = os.getenv("OPENAI_API_KEY")
            if not runtime_api_key:
                raise ValueError("OPENAI_API_KEY not found")
            log.info("✓ Using OPENAI_API_KEY from config")

    # Log runtime configuration
    endpoint_display = effective_base_url if effective_base_url else "api.openai.com"
    log.info(
        f"🤖 Agent execution - Model: {effective_model}, endpoint: {endpoint_display}, "
        f"top_k: {effective_top_k}, num_choices: {effective_num_choices}, excluded: {len(excluded or [])}"
    )

    # Create dynamic agent if needed
    needs_dynamic_agent = (
        (model and model != agent_model_config.name)
        or (base_url is not None and base_url != agent_model_config.base_url)
        or (runtime_api_key != api_key)  # API key mismatch - need new agent!
    )

    if needs_dynamic_agent:
        log.info(
            f"📦 Creating runtime agent with model={effective_model}, endpoint={effective_base_url or 'api.openai.com'}"
        )

        runtime_provider = OpenAIProvider(
            base_url=effective_base_url,
            api_key=runtime_api_key,
        )
        runtime_model = OpenAIChatModel(model_name=effective_model, provider=runtime_provider)

        agent_instance = Agent(
            model=runtime_model,
            system_prompt=get_agent_system_prompt(effective_num_choices),
            deps_type=AgentState,
        )

        # Register tools on the dynamic agent
        agent_instance.tool(search_tools, retries=2, prepare=cap_prepare)
        agent_instance.tool(rerank, retries=2, prepare=cap_prepare)
        agent_instance.tool(search_alternative, retries=2, prepare=cap_prepare)
        agent_instance.tool(repo_info, retries=2, prepare=cap_prepare)
        agent_instance.tool(run_example, retries=0, prepare=cap_prepare)

    elif num_choices is not None and num_choices != 3:
        # Model/base_url same but num_choices differs - create agent with updated prompt
        log.info(
            f"📦 Creating runtime agent with num_choices={effective_num_choices} (model: {effective_model})"
        )
        agent_instance = Agent(
            model=openai_model,
            system_prompt=get_agent_system_prompt(effective_num_choices),
            deps_type=AgentState,
        )

        # Register tools on the dynamic agent
        agent_instance.tool(search_tools, retries=2, prepare=cap_prepare)
        agent_instance.tool(rerank, retries=2, prepare=cap_prepare)
        agent_instance.tool(search_alternative, retries=2, prepare=cap_prepare)
        agent_instance.tool(repo_info, retries=2, prepare=cap_prepare)
        agent_instance.tool(run_example, retries=0, prepare=cap_prepare)

    else:
        log.info(f"♻️  Using global agent (model: {effective_model}, num_choices: {effective_num_choices})")

    log.debug(f"Prompt length: {len(prompt)} chars, has_image: {bool(image_paths)}")

    # ---- 5) Run the agent --------------------------------------------------
    result = agent_instance.run_sync(
        prompt,
        deps=deps,
        output_type=ToolSelection,
        usage_limits=UsageLimits(tool_calls_limit=20),
    ).output

    log.info(f"✅ Agent execution complete - choices returned: {len(result.choices)}")

    # ---- 6) Convert raw tool call records into ToolRunLog objects ----------
    for tc in getattr(deps, "tool_calls", []):
        tool_name = tc.get("tool")
        inputs = {k: v for k, v in tc.items() if k != "tool"}
        tool_logs.append(
            ToolRunLog(
                tool=tool_name,
                inputs=inputs,
            )
        )

    # ---- 7) Wrap into high-level AgentToolSelection ------------------------
    return AgentToolSelection(
        conversation=result.conversation,
        choices=result.choices,
        explanation=result.explanation,
        reason=result.reason,
        tool_calls=tool_logs,
    )


__all__ = ["run_agent", "agent"]