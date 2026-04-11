from __future__ import annotations

import os
import logging
import time
import asyncio
from datetime import datetime
from typing import List

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import BinaryContent

from ai_agent.generator.prompts import get_agent_system_prompt
from ai_agent.generator.schema import ToolSelection, Conversation, ConversationStatus
from ai_agent.utils.config import get_config
from .models import AgentToolSelection, ToolRunLog, UsageStats
from .tools.repo_info_tool import tool_repo_summary, RepoSummaryInput
from ai_agent.agent.utils import coerce_github_url_or_none
from .tools.search_tool import tool_search_tools, SearchToolsInput
from .tools.search_alternative_tool import (
    tool_search_alternative,
    SearchAlternativeInput,
)
from .tools.query_utils import sanitize_retrieval_query
from .utils import AgentState, limit_tool_calls, cap_prepare
from ai_agent.utils.image_meta import summarize_image_metadata, detect_ext_token

log = logging.getLogger("agent.core")

DEFAULT_NUM_CHOICES = int(os.getenv("NUM_CHOICES", "3"))

# ---------------------------------------------------------------------------
# Dynamic agent instance cache
# Key: (model_name, base_url, api_key_env, num_choices)
# Avoids rebuilding Agent/OpenAIProvider/model objects on every request when
# the UI repeatedly uses the same custom endpoint + model combination.
# ---------------------------------------------------------------------------
_AGENT_CACHE: dict[tuple, "Agent"] = {}

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
    log.info("Using OpenAIChatModel (chat/completions API) for custom endpoint")
    provider = OpenAIProvider(
        base_url=agent_model_config.base_url,
        api_key=api_key,
    )
    openai_model = OpenAIChatModel(
        model_name=agent_model_config.name,
        provider=provider,
    )
else:
    provider = OpenAIProvider(api_key=api_key)
    openai_model = OpenAIResponsesModel(
        model_name=agent_model_config.name,
        provider=provider,
    )

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
agent = Agent(
    model=openai_model,
    system_prompt=get_agent_system_prompt(DEFAULT_NUM_CHOICES),
    deps_type=AgentState,
    output_retries=int(os.getenv("AGENT_OUTPUT_RETRIES", "3")),
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

    effective_top_k = (
        ctx.deps.override_top_k if ctx.deps.override_top_k is not None else top_k
    )

    started = time.perf_counter()
    inp = SearchToolsInput(
        query=sanitize_retrieval_query(query),
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
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "original_formats": original_formats,
            "excluded": all_excluded,
            "timestamp": datetime.now().isoformat(),
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
    explicit_excluded = excluded or []
    global_excluded = getattr(ctx.deps, "excluded_tools", []) or []
    all_excluded = sorted(set(explicit_excluded + list(global_excluded)))

    original_formats = getattr(ctx.deps, "original_formats", []) or []
    image_paths = getattr(ctx.deps, "image_paths", []) or []

    started = time.perf_counter()
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
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "original_formats": original_formats,
            "excluded": all_excluded,
            "timestamp": datetime.now().isoformat(),
        }
    )

    return [c.model_dump(mode="python") for c in out.candidates]


@agent.tool(retries=2, prepare=cap_prepare)
@limit_tool_calls("repo_info_batch", cap=4)
async def repo_info_batch(
    ctx: RunContext[AgentState],
    urls: List[str],
) -> List[dict]:
    """Fetch repository summaries for multiple repositories in parallel."""
    started = time.perf_counter()

    if not urls:
        return []

    normalized: List[str] = []
    skipped: List[dict] = []
    seen: set[str] = set()
    for raw in urls:
        norm = coerce_github_url_or_none(raw)
        if not norm:
            skipped.append(
                {
                    "url": raw,
                    "skipped": True,
                    "reason": "NON_GITHUB_URL",
                }
            )
            continue
        if norm in seen:
            continue
        seen.add(norm)
        normalized.append(norm)

    tasks = [tool_repo_summary(RepoSummaryInput(url=u)) for u in normalized]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    results: List[dict] = []
    for url, outcome in zip(normalized, outcomes):
        if isinstance(outcome, Exception):
            results.append(
                {
                    "url": url,
                    "source": "error",
                    "error": str(outcome),
                }
            )
            continue
        payload = outcome.model_dump(mode="python")
        payload["url"] = url
        results.append(payload)

    if skipped:
        results.extend(skipped)

    ctx.deps.tool_calls.append(
        {
            "tool": "repo_info_batch",
            "requested": len(urls),
            "normalized": len(normalized),
            "returned": len(results),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "timestamp": datetime.now().isoformat(),
        }
    )

    return results


# ---------------------------------------------------------------------------
# High level entry point: run the agent on (text query + image)
# ---------------------------------------------------------------------------
def run_agent(
    task: str,
    image_paths: List[str],
    excluded: List[str] | None = None,
    conversation_history: List[str] | None = None,
    *,
    image_bytes: bytes | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
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

    IMPORTANT:
      The model only sees an actual image if `image_bytes` is provided.
      `image_paths` are used for metadata + tool context only.
    """
    run_started = time.perf_counter()
    if not image_paths:
        raise ValueError("run_agent requires at least one image path")

    tool_logs: List[ToolRunLog] = []

    # ---- 1) Derive image-based metadata and format hints --------------------
    metadata_started = time.perf_counter()
    meta_str = (
        image_metadata
        if image_metadata is not None
        else (summarize_image_metadata(image_paths) or "")
    )
    fmt_str = detect_ext_token(image_paths) or ""
    original_formats = [t.lower() for t in fmt_str.split()] if fmt_str else []
    metadata_duration_ms = round((time.perf_counter() - metadata_started) * 1000, 1)

    effective_top_k = top_k if top_k is not None else 12
    effective_num_choices = num_choices if num_choices is not None else 3

    # ---- 2) Prepare dependency state passed to all tools --------------------
    deps = AgentState(
        excluded_tools=excluded or [],
        override_model=model,
        override_base_url=base_url,
        override_top_k=effective_top_k,
        override_num_choices=effective_num_choices,
    )

    setattr(deps, "image_paths", list(image_paths))
    setattr(deps, "original_formats", original_formats)

    # ---- 3) Hidden metadata lines for the model ----------------------------
    hidden_meta = ""
    if original_formats:
        hidden_meta += "\n(Formats Hint: " + ",".join(original_formats) + ")"
    if meta_str:
        short_meta = " ".join(x.strip() for x in meta_str.splitlines() if x.strip())
        hidden_meta += (
            "\n(Image Metadata: "
            + short_meta[:500]
            + ("…" if len(short_meta) > 500 else "")
            + ")"
        )
    hidden_meta += f"\n(Search top_k: {effective_top_k})"

    extra_context = "\n\n**CRITICAL: Analyze the attached preview image showing the user's data.**\nUse visual observations (anatomy visible, image quality, dimensionality, contrast) combined with the metadata below to recommend tools. Reference what you see in your explanations."

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
    # Determine which agent instance to use
    # -----------------------------------------------------------------------
    agent_instance = agent
    effective_num_choices = num_choices if num_choices is not None else 3
    effective_model = model if model else agent_model_config.name
    effective_top_k = top_k if top_k is not None else 12

    # When model is provided from UI, base_url comes with it (can be None for OpenAI)
    if model:
        # Use api_key_env from config if provided, otherwise default to OPENAI_API_KEY
        key_env_name = api_key_env if api_key_env else "OPENAI_API_KEY"
        runtime_api_key = os.getenv(key_env_name)
        if not runtime_api_key:
            raise ValueError(
                f"{key_env_name} not found in environment. Cannot use this model."
            )
        effective_base_url = base_url  # Can be None for OpenAI
        log.info(f"✓ Using {key_env_name} for model {effective_model}")
        log.debug(f"{key_env_name} is set: {bool(runtime_api_key)}")
    else:
        # No model override - use config defaults
        effective_base_url = agent_model_config.base_url
        runtime_api_key = api_key  # Already loaded from config at startup
        log.info(f"✓ Using API key from config for model {effective_model}")

    # Log runtime configuration
    endpoint_display = effective_base_url if effective_base_url else "api.openai.com"
    log.info(
        f"🤖 Agent execution - Model: {effective_model}, endpoint: {endpoint_display}, "
        f"top_k: {effective_top_k}, num_choices: {effective_num_choices}, excluded: {len(excluded or [])}"
    )

    needs_dynamic_agent = model is not None

    if needs_dynamic_agent:
        cache_key = (effective_model, effective_base_url or "", api_key_env or "OPENAI_API_KEY", effective_num_choices)
        agent_instance = _AGENT_CACHE.get(cache_key)
        if agent_instance is None:
            log.info(
                f"📦 Creating runtime agent with model={effective_model}, endpoint={effective_base_url or 'api.openai.com'}"
            )

            runtime_provider = OpenAIProvider(
                base_url=effective_base_url,
                api_key=runtime_api_key,
            )

            # Use OpenAIChatModel (chat/completions) for custom endpoints, OpenAIResponsesModel for default OpenAI
            if effective_base_url:
                log.info("Using OpenAIChatModel (chat/completions API) for custom endpoint")
                runtime_model = OpenAIChatModel(
                    model_name=effective_model, provider=runtime_provider
                )
            else:
                runtime_model = OpenAIResponsesModel(
                    model_name=effective_model, provider=runtime_provider
                )

            agent_instance = Agent(
                model=runtime_model,
                system_prompt=get_agent_system_prompt(effective_num_choices),
                deps_type=AgentState,
                output_retries=int(os.getenv("AGENT_OUTPUT_RETRIES", "3")),
            )

            # Register tools on the dynamic agent
            agent_instance.tool(search_tools, retries=2, prepare=cap_prepare)
            agent_instance.tool(search_alternative, retries=2, prepare=cap_prepare)
            agent_instance.tool(repo_info_batch, retries=2, prepare=cap_prepare)

            _AGENT_CACHE[cache_key] = agent_instance
        else:
            log.info(
                f"♻️  Reusing cached dynamic agent (model: {effective_model}, num_choices: {effective_num_choices})"
            )

    elif (
        num_choices is not None and num_choices != DEFAULT_NUM_CHOICES
    ):
        cache_key = (effective_model, effective_base_url or "", api_key_env or "OPENAI_API_KEY", effective_num_choices)
        agent_instance = _AGENT_CACHE.get(cache_key)
        if agent_instance is None:
            log.info(
                f"📦 Creating runtime agent with num_choices={effective_num_choices} (model: {effective_model})"
            )
            agent_instance = Agent(
                model=openai_model,
                system_prompt=get_agent_system_prompt(effective_num_choices),
                deps_type=AgentState,
                output_retries=int(os.getenv("AGENT_OUTPUT_RETRIES", "3")),
            )

            # Register tools on the dynamic agent
            agent_instance.tool(search_tools, retries=2, prepare=cap_prepare)
            agent_instance.tool(search_alternative, retries=2, prepare=cap_prepare)
            agent_instance.tool(repo_info_batch, retries=2, prepare=cap_prepare)

            _AGENT_CACHE[cache_key] = agent_instance
        else:
            log.info(
                f"♻️  Reusing cached dynamic agent with num_choices={effective_num_choices} (model: {effective_model})"
            )

    else:
        log.info(
            f"♻️  Using global agent (model: {effective_model}, num_choices: {effective_num_choices})"
        )

    log.debug(
        f"Prompt length: {len(prompt)} chars, has_image_paths: {bool(image_paths)}, has_image_bytes: {bool(image_bytes)}"
    )

    # ---- 5) Build multimodal prompt if image bytes provided ----------------
    if image_bytes:
        log.info(
            f"🖼️  Sending image preview to model ({len(image_bytes)} bytes = {len(image_bytes)/1024:.1f}KB)"
        )
        user_prompt = [
            prompt,
            BinaryContent(
                data=image_bytes,
                media_type="image/png",
            ),
        ]
    else:
        log.warning(
            "⚠️  No image bytes provided - the model will not see the image preview"
        )
        user_prompt = prompt

    # ---- 6) Run the agent --------------------------------------------------
    try:
        llm_started = time.perf_counter()
        run_result = agent_instance.run_sync(
            user_prompt,
            deps=deps,
            output_type=ToolSelection,
            usage_limits=UsageLimits(tool_calls_limit=20),
        )
        llm_duration_ms = round((time.perf_counter() - llm_started) * 1000, 1)
        result = run_result.output

        log.info(
            f"✅ Agent execution complete - choices returned: {len(result.choices)}"
        )

        # Log usage (helpful, but may not explicitly expose image-specific counters)
        if run_result.usage:
            usage = run_result.usage()
            log.info(
                f"📊 Usage: total_tokens={usage.total_tokens}, "
                f"input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}"
            )

        # Warn if using non-OpenAI endpoint with images
        if image_bytes and effective_base_url:
            log.warning(
                "⚠️  Using custom endpoint - confirm the selected model supports vision."
            )

    except Exception as e:
        # Handle global tool quota limit (UsageLimitExceeded) and other errors gracefully
        error_msg = str(e)
        llm_duration_ms = round((time.perf_counter() - llm_started) * 1000, 1)
        log.warning(f"⚠️  Agent execution encountered an error: {error_msg}")
        run_result = None  # Ensure run_result is defined for usage stats extraction

        # Check if this is a usage limit error (global tool quota)
        if (
            "UsageLimitExceeded" in str(type(e).__name__)
            or "tool_calls_limit" in error_msg.lower()
        ):
            log.warning(
                "Global tool call quota reached - continuing with partial results"
            )

            result = ToolSelection(
                conversation=Conversation(
                    status=ConversationStatus.COMPLETE,
                    context="The agent reached the maximum number of tool calls allowed. Please try a more specific query or break down your request into smaller parts.",
                    question=None,
                    options=None,
                ),
                choices=[],
                explanation="Tool call limit reached during execution. Try refining your query.",
                reason=None,
            )
        else:
            raise

    # ---- 7) Convert raw tool call records into ToolRunLog objects ----------
    for tc in getattr(deps, "tool_calls", []):
        tool_name = tc.get("tool")
        timestamp = tc.get("timestamp")
        error = tc.get("error")
        inputs = {
            k: v for k, v in tc.items() if k not in ("tool", "timestamp", "error")
        }
        tool_logs.append(
            ToolRunLog(
                tool=tool_name,
                inputs=inputs,
                timestamp=timestamp,
                error=error,
            )
        )

    stage_counts: dict[str, int] = {}
    stage_durations: dict[str, float] = {}
    for tc in getattr(deps, "tool_calls", []):
        name = tc.get("tool", "unknown")
        stage_counts[name] = stage_counts.get(name, 0) + 1
        duration_ms = tc.get("duration_ms")
        if isinstance(duration_ms, (int, float)):
            stage_durations[name] = stage_durations.get(name, 0.0) + float(duration_ms)

    total_duration_ms = round((time.perf_counter() - run_started) * 1000, 1)
    log.info(
        "⏱️ Latency summary: total_ms=%s metadata_ms=%s llm_ms=%s tools=%s tool_ms=%s",
        total_duration_ms,
        metadata_duration_ms,
        llm_duration_ms,
        stage_counts,
        {k: round(v, 1) for k, v in stage_durations.items()},
    )

    # ---- 8) Extract usage statistics if available -------------------------
    usage_stats = None
    if run_result and hasattr(run_result, "usage") and run_result.usage:
        usage = run_result.usage()
        usage_stats = UsageStats(
            total_tokens=usage.total_tokens,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    # ---- 9) Wrap into high-level AgentToolSelection ------------------------
    return AgentToolSelection(
        conversation=result.conversation,
        choices=result.choices,
        explanation=result.explanation,
        reason=result.reason,
        tool_calls=tool_logs,
        usage=usage_stats,
    )


__all__ = ["run_agent", "agent"]
