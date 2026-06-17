"""Chat orchestration service.

This is the transport-agnostic version of the orchestration that used to live
inside ``ai_agent.ui.handlers.respond``. It owns:

  - validating/registering uploaded assets on the session
  - resolving asset previews into bytes for the VLM
  - resolving model/top_k/num_choices overrides from the UI
  - invoking the agent
  - turning the agent's structured output into a ``ChatTurnResult`` that
    contains both a markdown-friendly text and the raw recommendation list
  - handling pending tool approvals and demo confirmations

Phase-1 entrypoint is synchronous (``process_turn``). The FastAPI router in
phase 2 wraps this in an SSE stream, emitting recommendations, tool traces
and pending actions as discrete events.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ai_agent.agent.agent import run_agent
from ai_agent.agent.tools.mcp import (
    alias_is_tool_level,
    extract_downloads,
    extract_metadata,
    extract_output_field,
    extract_preview,
    get_tool,
    list_tool_endpoints,
    resolve_catalog_alias,
)
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.utils.tags import parse_exclusions, strip_tags
from ai_agent.utils.utils import _is_affirmative

from .files import asset_paths, ingest_files
from .sessions import Asset, Session

log = logging.getLogger("services.chat")


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------
TurnStatus = Literal[
    "ok", "needs_clarification", "no_results", "error", "pending_action", "tool_executed"
]


@dataclass
class Recommendation:
    rank: int
    name: str
    accuracy: float
    why: str
    doc: Optional[Dict[str, Any]] = None
    demo_url: Optional[str] = None


@dataclass
class PendingAction:
    """A turn that ends asking the user to confirm something.

    The client surfaces an Approve / Decline control; calling
    ``approve_pending`` or ``decline_pending`` on the service resumes the
    flow.
    """

    type: Literal["demo_confirm", "tool_approval"]
    tool_name: str
    display_name: Optional[str] = None
    icon: Optional[str] = None
    image_name: Optional[str] = None
    demo_url: Optional[str] = None
    prompt: str = ""
    endpoint_id: Optional[str] = None
    endpoint_display_name: Optional[str] = None
    recommendation_name: Optional[str] = None
    recommendation_rank: Optional[int] = None
    matched_alias: Optional[str] = None
    api_name: Optional[str] = None
    required_inputs: List[str] = field(default_factory=list)


@dataclass
class Clarification:
    question: str
    context: Optional[str] = None
    options: List[str] = field(default_factory=list)


@dataclass
class EndpointSelection:
    tool: Any
    score: float
    reason: str = ""


@dataclass
class ChatTurnResult:
    status: TurnStatus
    text: str = ""
    recommendations: List[Recommendation] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    pending_action: Optional[PendingAction] = None
    clarification: Optional[Clarification] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # Extras used by Gradio rendering (preview images, downloads from a
    # tool execution turn). Empty for normal chat turns.
    images: List[str] = field(default_factory=list)
    files: List[tuple] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------
@dataclass
class ChatRequest:
    message: str = ""
    asset_ids: List[str] = field(default_factory=list)
    """Asset ids previously registered via ``services.files.ingest_files``."""

    new_file_paths: List[str] = field(default_factory=list)
    """Convenience for the Gradio adapter — files uploaded as part of this
    turn that haven't been pre-ingested. The chat service will ingest them
    and merge their asset_ids into ``asset_ids``."""

    model: Optional[str] = None
    """Display name from ``config.yaml#available_models`` (e.g.
    ``"openai/gpt-oss-120b [EPFL]"``). Resolved against the config map."""

    top_k: Optional[int] = None
    num_choices: Optional[int] = None


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------
def process_turn(
    session: Session,
    request: ChatRequest,
    doc_index: Dict[str, SoftwareDoc],
) -> ChatTurnResult:
    """Run one user turn end-to-end against ``session``.

    Mutates the session in place (history, last_asset_ids, banlist,
    pending_*, tool_calls). Returns a structured result the transport layer
    can render or stream.
    """
    session.touch()

    # 1) Ingest any newly-attached files (Gradio gives us raw paths)
    if request.new_file_paths:
        result = ingest_files(session, request.new_file_paths)
        if result.validation_errors:
            issues = "\n".join(f"• {x}" for x in result.validation_errors)
            text = f"⚠️ File validation issues:\n\n{issues}"
            session.conversation_history.append(f"Assistant: {text}")
            return ChatTurnResult(status="error", text=text, error="invalid_files")
        # Merge ingested asset_ids with any client-supplied ones
        new_ids = [a.asset_id for a in result.assets]
        request.asset_ids = list(dict.fromkeys(request.asset_ids + new_ids))

    # 2) Reject empty turns (no message AND no attachments)
    has_text = bool((request.message or "").strip())
    has_attachments = bool(request.asset_ids or session.last_asset_ids)
    if not has_text and not has_attachments:
        text = "Please provide a message or upload files."
        return ChatTurnResult(status="error", text=text, error="empty_input")

    # 3) Parse banlist tags out of the message
    clean_message = strip_tags(request.message or "")
    session.banlist |= set(parse_exclusions(request.message or ""))
    session.conversation_history.append(f"User: {clean_message}")

    # 4) Demo confirmation short-circuit
    if session.pending_demo_tool and _is_affirmative(request.message):
        return _execute_pending_demo(session, request.asset_ids)

    if session.pending_demo_tool:
        # Anything that isn't affirmative cancels the pending demo.
        _clear_pending(session)
    elif session.pending_tool_approval:
        # A fresh user turn replaces stale approval state.
        _clear_pending(session)

    # 5) Resolve attachment paths (default to last upload if none provided)
    if request.asset_ids:
        # Anything explicitly attached becomes the "active" set
        effective_paths, attached_assets = asset_paths(session, request.asset_ids)
        session.last_asset_ids = [a.asset_id for a in attached_assets]
    else:
        effective_paths = session.last_asset_paths()
        attached_assets = [session.assets[a] for a in session.last_asset_ids if a in session.assets]

    # Images are optional. When the user hasn't uploaded anything we run the
    # agent in text-only mode — retrieval still works on the prompt.

    # 6) Find the latest preview asset (used as the VLM image)
    preview_asset = session.last_preview()
    image_bytes: Optional[bytes] = None
    if preview_asset and preview_asset.preview_path:
        try:
            preview_path = Path(preview_asset.preview_path)
            if preview_path.exists():
                image_bytes = preview_path.read_bytes()
        except Exception as e:
            log.warning("Failed to read preview bytes: %r", e)

    image_metadata = preview_asset.metadata_text if preview_asset else None

    # 7) Resolve model config from the display name (if provided by the UI)
    model_name, base_url_override, api_key_env = _resolve_model_choice(request.model)

    # 8) Run the agent
    log.info(
        "Running agent: task=%r, attachments=%d, excluded=%d, model=%s",
        clean_message,
        len(effective_paths),
        len(session.banlist),
        request.model,
    )

    try:
        agent_result = run_agent(
            clean_message,
            image_paths=effective_paths,
            image_bytes=image_bytes,
            excluded=list(session.banlist),
            conversation_history=session.conversation_history,
            model=model_name,
            base_url=base_url_override if request.model else None,
            api_key_env=api_key_env,
            top_k=request.top_k,
            num_choices=request.num_choices,
            image_metadata=image_metadata,
        )
    except ValueError as e:
        return _format_config_error(e, session)
    except Exception as e:
        return _format_runtime_error(e, session)

    return _shape_agent_result(
        session, agent_result, doc_index, effective_paths, clean_message
    )


def approve_pending(session: Session) -> ChatTurnResult:
    """Resume a turn that ended with a ``tool_approval`` pending action.

    Calls the registered tool with the previously-captured parameters, then
    clears the pending state.
    """
    tool_name = session.pending_tool_approval
    if not tool_name:
        return ChatTurnResult(
            status="error",
            text="There is no pending tool approval to confirm.",
            error="no_pending_action",
        )
    params = dict(session.pending_tool_params)
    return _execute_registered_tool(session, tool_name, params)


def decline_pending(session: Session) -> ChatTurnResult:
    """Decline both pending demo and pending tool approval."""
    session.pending_demo_tool = None
    session.pending_demo_url = None
    _clear_pending(session)
    text = "👍 Got it — I won't run that. Tell me what to try instead."
    session.conversation_history.append(f"Assistant: {text}")
    return ChatTurnResult(status="ok", text=text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_model_choice(display_name: Optional[str]):
    if not display_name:
        return None, None, None
    # Lazy import to avoid the Gradio dependency footprint when running from
    # the FastAPI backend.
    try:
        from ai_agent.ui.components import get_model_config

        cfg = get_model_config(display_name)
    except Exception as e:
        log.warning("Could not resolve model %r: %r", display_name, e)
        return None, None, None
    return (
        cfg.get("name"),
        cfg.get("base_url"),
        cfg.get("api_key_env", "OPENAI_API_KEY"),
    )


def _format_config_error(exc: ValueError, session: Session) -> ChatTurnResult:
    msg = str(exc)
    log.error("Configuration error: %s", msg)
    text = f"⚠️ **Configuration Error**\n\n{msg}\n\n"
    if "EPFL_API_KEY" in msg:
        text += (
            "💡 **Tip:** EPFL models require VPN connection and `EPFL_API_KEY` in "
            "your `.env` file. Try selecting an OpenAI model instead."
        )
    elif "OPENAI_API_KEY" in msg:
        text += "💡 **Tip:** Set `OPENAI_API_KEY` in your `.env` file to use OpenAI models."
    session.conversation_history.append(f"Assistant: {text}")
    return ChatTurnResult(status="error", text=text, error=msg)


def _format_runtime_error(exc: Exception, session: Session) -> ChatTurnResult:
    msg = str(exc)
    log.error("Agent execution error: %s", msg, exc_info=True)
    text = f"❌ **Error**\n\n{msg}\n\n"
    if "key_model_access_denied" in msg or "key not allowed" in msg.lower():
        text += "💡 **Tip:** This API key doesn't have access to this model.\n\n"
    elif "ConnectError" in msg or "Connection" in msg:
        text += "💡 **Tip:** Connection failed. If using EPFL models, ensure you're on EPFL VPN."
    session.conversation_history.append(f"Assistant: {text}")
    return ChatTurnResult(status="error", text=text, error=msg)


def _shape_agent_result(
    session: Session,
    agent_result,
    doc_index: Dict[str, SoftwareDoc],
    effective_paths: List[str],
    request_text: str = "",
) -> ChatTurnResult:
    """Translate an ``AgentToolSelection`` into a ``ChatTurnResult``."""
    legacy = agent_result.to_legacy_dict()

    tool_traces = legacy.get("tool_calls", []) or []
    if tool_traces:
        session.tool_calls.extend(tool_traces)

    usage = legacy.get("usage")
    usage_payload = None
    if usage:
        usage_payload = {
            "total": usage.get("total_tokens", 0),
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
        }

    status = legacy["conversation"]["status"]
    if status == "needs_clarification":
        question = legacy["conversation"]["question"]
        context = legacy["conversation"].get("context")
        options = legacy["conversation"].get("options", []) or []

        text = f"ℹ️ **I need more information:**\n\n{question}\n\n"
        if options:
            text += "**Options:**\n" + "\n".join(f"- {o}" for o in options) + "\n\n"
        if context:
            text += f"_{context}_"
        session.conversation_history.append(f"Assistant: {text}")
        return ChatTurnResult(
            status="needs_clarification",
            text=text,
            tool_traces=tool_traces,
            usage=usage_payload,
            clarification=Clarification(
                question=question, context=context, options=options
            ),
        )

    choices = legacy.get("choices") or []
    if not choices:
        reason = legacy.get("reason") or ""
        explanation = legacy.get("explanation") or ""
        parts = ["❌ **No suitable tools found.**\n"]
        if reason:
            parts.append(f"**Reason:** `{reason}`\n")
        if explanation:
            parts.append(explanation)
        text = "\n".join(parts)
        session.conversation_history.append(f"Assistant: {text}")
        return ChatTurnResult(
            status="no_results",
            text=text,
            tool_traces=tool_traces,
            usage=usage_payload,
        )

    # Recommendations path. New recommendation results replace stale pending actions.
    _clear_pending(session)
    session.last_choices = {c["name"]: c for c in choices}
    for c in choices:
        if c.get("name"):
            session.banlist.add(c["name"])

    recommendations: List[Recommendation] = []
    for i, c in enumerate(choices, 1):
        doc = doc_index.get(c["name"])
        recommendations.append(
            Recommendation(
                rank=i,
                name=c["name"],
                accuracy=float(c.get("accuracy", 0.0)),
                why=c.get("why", ""),
                doc=doc.model_dump(mode="python") if doc is not None else None,
                demo_url=c.get("demo_link"),
            )
        )

    top = choices[0]
    text_parts = [
        f"✅ **I recommend {top['name']}** ({top.get('accuracy', 0):.1f}% match)\n",
        f"_{top.get('why', '')}_\n",
    ]
    text = "\n".join(text_parts)

    pending_action = _select_pending_action(
        session, choices, effective_paths, request_text
    )
    if pending_action and pending_action.recommendation_rank and pending_action.recommendation_rank > 1:
        text += (
            f"\n\nA runnable demo is available for rank {pending_action.recommendation_rank}, "
            f"{pending_action.recommendation_name}, because higher-ranked recommendations do not have an available configured Gradio endpoint."
        )

    session.conversation_history.append(f"Assistant: {text}")
    return ChatTurnResult(
        status="pending_action" if pending_action else "ok",
        text=text,
        recommendations=recommendations,
        tool_traces=tool_traces,
        pending_action=pending_action,
        usage=usage_payload,
    )


def _clear_pending(session: Session) -> None:
    session.pending_demo_tool = None
    session.pending_demo_url = None
    session.pending_tool_approval = None
    session.pending_tool_endpoint = None
    session.pending_recommendation_name = None
    session.pending_recommendation_rank = None
    session.pending_catalog_alias = None
    session.pending_tool_params = {}


def _select_pending_action(
    session: Session,
    choices: List[Dict[str, Any]],
    effective_paths: List[str],
    request_text: str = "",
) -> Optional[PendingAction]:
    image_path = effective_paths[0] if effective_paths else None
    for rank, choice in enumerate(choices, 1):
        alias = choice.get("name") or ""
        tool_config = resolve_catalog_alias(alias)
        if not tool_config or not tool_config.endpoint:
            continue
        endpoint_selection = _select_endpoint_for_choice(
            tool_config=tool_config,
            alias=alias,
            choice=choice,
            request_text=request_text,
            file_count=len(effective_paths),
        )
        tool_config = endpoint_selection.tool
        if not tool_config.is_runnable():
            continue
        required_inputs = [
            p.name
            for p in tool_config.endpoint.input_mapping.parameters
            if p.required and p.source in ("session_file", "image_path")
        ]
        if required_inputs and not image_path:
            continue
        endpoint_id = tool_config.endpoint.id
        session.pending_tool_approval = tool_config.name
        session.pending_tool_endpoint = endpoint_id
        session.pending_recommendation_name = alias
        session.pending_recommendation_rank = rank
        session.pending_catalog_alias = alias
        session.pending_tool_params = {
            "endpoint_id": endpoint_id,
            "image_path": image_path,
            "image_paths": list(effective_paths),
            "description": f"Recommended by agent: {choice.get('why', '')}",
        }
        approval = tool_config.endpoint.approval
        prompt = approval.message or f"Run {tool_config.endpoint.display_name} on your image?"
        return PendingAction(
            type="tool_approval",
            tool_name=tool_config.name,
            display_name=tool_config.display_name,
            icon=tool_config.icon,
            image_name=os.path.basename(image_path) if image_path else None,
            demo_url=tool_config.gradio_url,
            prompt=prompt,
            endpoint_id=endpoint_id,
            endpoint_display_name=tool_config.endpoint.display_name,
            recommendation_name=alias,
            recommendation_rank=rank,
            matched_alias=alias,
            api_name=tool_config.api_name,
            required_inputs=required_inputs,
        )
    return None


def _select_endpoint_for_choice(
    *,
    tool_config,
    alias: str,
    choice: Dict[str, Any],
    request_text: str,
    file_count: int,
) -> EndpointSelection:
    if not alias_is_tool_level(alias, tool_config):
        return EndpointSelection(tool=tool_config, score=999.0, reason="endpoint alias")

    candidates = [
        t
        for t in list_tool_endpoints(tool_config.name)
        if t.endpoint and t.is_runnable() and _required_file_count(t) <= file_count
    ]
    if len(candidates) <= 1:
        return EndpointSelection(
            tool=candidates[0] if candidates else tool_config,
            score=0.0,
            reason="single endpoint",
        )

    text = " ".join(
        str(x or "")
        for x in (
            request_text,
            choice.get("why"),
            choice.get("context"),
            choice.get("name"),
        )
    )
    scored = sorted(
        (
            EndpointSelection(
                tool=candidate,
                score=_score_endpoint(candidate, text, file_count),
                reason="request match",
            )
            for candidate in candidates
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    best = scored[0]
    runner_up = scored[1].score if len(scored) > 1 else 0.0
    default_id = tool_config.endpoint_id

    if best.score >= 2.0 and best.score - runner_up >= 0.75:
        return best
    if default_id:
        default = next((c for c in candidates if c.endpoint_id == default_id), None)
        if default:
            return EndpointSelection(tool=default, score=0.0, reason="default endpoint")
    return best


def _score_endpoint(tool_config, request_text: str, file_count: int) -> float:
    endpoint = tool_config.endpoint
    if not endpoint:
        return 0.0
    text = _normalize_match_text(request_text)
    corpus = _endpoint_match_corpus(tool_config)
    score = 0.0

    request_tokens = _tokenize_for_match(text)
    corpus_tokens = _tokenize_for_match(corpus)
    score += len(request_tokens & corpus_tokens) * 0.4

    for alias in endpoint.catalog_aliases:
        alias_text = _normalize_match_text(alias)
        if alias_text and alias_text in text:
            score += 3.0

    endpoint_key = _normalize_match_text(
        " ".join([endpoint.id, endpoint.display_name, endpoint.description or ""])
    )
    if any(p in text for p in ("frame to frame", "frame-to-frame", "moving frame")):
        if "frame" in endpoint_key:
            score += 4.0
        if "stack" in endpoint_key and "frame" not in endpoint.id:
            score -= 1.0

    if any(
        p in text
        for p in (
            "reference stack",
            "moving stack",
            "separate reference",
            "external reference",
            "stack to stack",
            "stack-to-stack",
        )
    ):
        if "reference" in endpoint_key or "stack to stack" in endpoint_key:
            score += 4.0
        if "intra" in endpoint_key:
            score -= 2.0

    if any(
        p in text
        for p in (
            "same stack",
            "single stack",
            "within stack",
            "within the stack",
            "intra stack",
            "intra-stack",
            "stabilize stack",
            "drift correct",
            "drift correction",
        )
    ):
        if "intra" in endpoint_key or "within" in endpoint_key:
            score += 4.0
        if "reference" in endpoint_key and file_count < 2:
            score -= 1.0

    if file_count >= 2:
        if _required_file_count(tool_config) >= 2:
            score += 1.5
    elif _required_file_count(tool_config) > file_count:
        score -= 5.0

    return score


def _endpoint_match_corpus(tool_config) -> str:
    endpoint = tool_config.endpoint
    gradio = tool_config.gradio
    if not endpoint:
        return ""
    parts: List[str] = [
        tool_config.name,
        tool_config.display_name,
        endpoint.id,
        endpoint.display_name,
        endpoint.description or "",
        endpoint.api_name or "",
        " ".join(endpoint.catalog_aliases),
    ]
    if gradio:
        parts.extend([gradio.description or "", " ".join(gradio.catalog_aliases)])
    for param in endpoint.input_mapping.parameters:
        parts.extend([param.name, param.source, str(param.param or ""), str(param.value or "")])
    return _normalize_match_text(" ".join(parts))


def _required_file_count(tool_config) -> int:
    endpoint = tool_config.endpoint
    if not endpoint:
        return 0
    required_indices = [
        p.file_index
        for p in endpoint.input_mapping.parameters
        if p.required and p.source in ("session_file", "image_path")
    ]
    return max(required_indices) + 1 if required_indices else 0


def _normalize_match_text(value: str) -> str:
    return " ".join((value or "").replace("_", " ").replace("-", " ").casefold().split())


def _tokenize_for_match(value: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "for",
        "from",
        "in",
        "my",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
    return {t for t in re.findall(r"[a-z0-9]+", value) if len(t) > 2 and t not in stop}


def _execute_pending_demo(session: Session, attached_ids: List[str]) -> ChatTurnResult:
    """Legacy generic-demo state is no longer executable outside configured Gradio endpoints."""
    _clear_pending(session)
    text = "No configured runnable Gradio endpoint is pending."
    session.conversation_history.append(f"Assistant: {text}")
    return ChatTurnResult(status="ok", text=text)


def _execute_registered_tool(
    session: Session, tool_name: str, params: Dict[str, Any]
) -> ChatTurnResult:
    """Execute a registered tool that gated on user approval."""
    endpoint_for_execution = params.get("endpoint_id") or session.pending_tool_endpoint
    tool_config = get_tool(tool_name, endpoint_for_execution) if endpoint_for_execution else get_tool(tool_name)
    if not tool_config:
        text = f"❌ Error: Unknown tool '{tool_name}'"
        _clear_pending(session)
        session.conversation_history.append(f"Assistant: {text}")
        return ChatTurnResult(status="error", text=text, error="unknown_tool")

    started = time.time()
    text = f"{tool_config.icon} Running {tool_config.display_name}...\n\n"
    images: List[str] = []
    files: List[tuple] = []
    artifact_assets: Dict[str, Asset] = {}
    try:
        # Backfill missing image path from last upload
        if "image_path" in params and not params["image_path"]:
            paths = session.last_asset_paths()
            if paths:
                params["image_path"] = paths[0]

        endpoint_id = endpoint_for_execution or tool_config.endpoint_id
        input_obj = tool_config.input_model(
            tool_id=tool_config.name,
            endpoint_id=endpoint_id,
            image_path=params.get("image_path"),
            image_paths=params.get("image_paths", []),
            description=params.get("description"),
            params=params.get("params", {}),
        )
        result = tool_config.executor(input_obj)

        success = extract_output_field(result, tool_config.success_field)
        error = extract_output_field(result, tool_config.error_field)
        compute_time_seconds = (
            extract_output_field(result, tool_config.compute_time_field) or 0.0
        )
        notes = extract_output_field(result, tool_config.notes_field)

        session.tool_calls.append(
            {
                "tool": tool_name,
                "endpoint": params.get("endpoint_id") or session.pending_tool_endpoint,
                "recommendation": session.pending_recommendation_name,
                "recommendation_rank": session.pending_recommendation_rank,
                "matched_alias": session.pending_catalog_alias,
                "success": success,
                "compute_time_seconds": compute_time_seconds,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                **params,
            }
        )

        if success:
            text += f"✅ {tool_config.display_name} completed!\n\n"
            preview_path = extract_preview(result, tool_name)
            if preview_path and os.path.exists(preview_path):
                asset = _register_tool_artifact(session, preview_path)
                if asset and asset.preview_path:
                    artifact_assets[preview_path] = asset
                    images.append(_asset_preview_url(asset))
            for dp in extract_downloads(result, tool_name):
                if os.path.exists(dp):
                    asset = artifact_assets.get(dp) or _register_tool_artifact(session, dp)
                    if asset:
                        artifact_assets[dp] = asset
                        files.append(
                            (
                                _asset_raw_url(asset),
                                f"Download {tool_config.display_name} result",
                            )
                        )
            metadata = extract_metadata(result, tool_name)
            if metadata:
                text += f"_{metadata}_\n\n"
            if notes:
                text += f"_{notes}_\n\n"
        else:
            text += f"❌ {tool_config.display_name} failed.\n\n"
            if error:
                text += f"**Error:** {error}\n\n"
    except Exception as e:
        log.exception("Tool %s execution failed", tool_name)
        text += f"❌ Error: {e}\n\n"

    _clear_pending(session)
    elapsed = time.time() - started
    log.info("Tool %s finished in %.2fs", tool_name, elapsed)
    session.conversation_history.append(f"Assistant: {text}")
    return ChatTurnResult(
        status="tool_executed", text=text, images=images, files=files
    )


def _register_tool_artifact(session: Session, path: str) -> Optional[Asset]:
    """Register a tool output for API serving without changing active inputs."""
    previous_last_asset_ids = list(session.last_asset_ids)
    result = None
    try:
        result = ingest_files(session, [path])
    except Exception:
        log.exception("Tool artifact registration failed for %s", path)
        return None
    finally:
        session.last_asset_ids = previous_last_asset_ids
        session.touch()
    if result.validation_errors:
        log.warning(
            "Tool artifact validation failed for %s: %s",
            path,
            result.validation_errors,
        )
    return result.assets[0] if result.assets else None


def _asset_preview_url(asset: Asset) -> str:
    return f"/api/files/preview/{asset.asset_id}"


def _asset_raw_url(asset: Asset) -> str:
    return f"/api/files/asset/{asset.asset_id}/raw"


__all__ = [
    "ChatRequest",
    "ChatTurnResult",
    "Clarification",
    "PendingAction",
    "Recommendation",
    "approve_pending",
    "decline_pending",
    "process_turn",
]
