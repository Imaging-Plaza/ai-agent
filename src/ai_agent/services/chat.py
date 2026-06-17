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
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ai_agent.agent.agent import run_agent
from ai_agent.agent.tools.gradio_space_tool import RunExampleInput, tool_run_example
from ai_agent.agent.tools.mcp import (
    extract_downloads,
    extract_metadata,
    extract_output_field,
    extract_preview,
    get_tool,
)
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.utils.tags import parse_exclusions, strip_tags
from ai_agent.utils.utils import _is_affirmative

from .files import asset_paths, ingest_files
from .sessions import Session

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


@dataclass
class Clarification:
    question: str
    context: Optional[str] = None
    options: List[str] = field(default_factory=list)


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
        # Anything that isn't affirmative cancels the pending demo
        session.pending_demo_tool = None
        session.pending_demo_url = None

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

    return _shape_agent_result(session, agent_result, doc_index, effective_paths)


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
    session.pending_tool_approval = None
    session.pending_tool_params = {}
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

    # Recommendations path
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

    # Decide whether the top tool implies a pending action (registry-driven)
    pending_action: Optional[PendingAction] = None
    top_name = top["name"]
    demo_url = top.get("demo_link") or ""
    tool_config = get_tool(top_name)

    if tool_config and tool_config.requires_approval:
        image_path = effective_paths[0] if effective_paths else None
        session.pending_tool_approval = tool_config.name
        session.pending_tool_params = {
            "image_path": image_path,
            "description": f"Recommended by agent: {top.get('why', '')}",
        }
        pending_action = PendingAction(
            type="tool_approval",
            tool_name=tool_config.name,
            display_name=tool_config.display_name,
            icon=tool_config.icon,
            image_name=os.path.basename(image_path) if image_path else None,
            demo_url=demo_url or None,
            prompt=f"Run {tool_config.display_name} on your image?",
        )
    elif demo_url:
        session.pending_demo_tool = top_name
        session.pending_demo_url = demo_url
        pending_action = PendingAction(
            type="demo_confirm",
            tool_name=top_name,
            demo_url=demo_url,
            prompt=f"Would you like me to run the demo for {top_name}?",
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


def _execute_pending_demo(session: Session, attached_ids: List[str]) -> ChatTurnResult:
    """Generic-demo flow (no registry entry, just a runnable demo URL)."""
    tool_name = session.pending_demo_tool
    demo_url = session.pending_demo_url
    log.info("User confirmed demo run for %s", tool_name)

    candidate_paths: List[str]
    if attached_ids:
        candidate_paths, _ = asset_paths(session, attached_ids)
    else:
        candidate_paths = session.last_asset_paths()

    session.pending_demo_tool = None
    session.pending_demo_url = None

    if not candidate_paths:
        text = "⚠️ No files available. Please upload an image first."
        session.conversation_history.append(f"Assistant: {text}")
        return ChatTurnResult(status="error", text=text, error="no_attachments")

    # Prefer TIFF if any
    pick = next(
        (
            p
            for p in candidate_paths
            if os.path.splitext(p)[1].lower() in (".tif", ".tiff")
        ),
        candidate_paths[0],
    )

    text = f"🚀 Running demo for **{tool_name}**...\n\n"
    images: List[str] = []
    files: List[tuple] = []
    try:
        demo_result = tool_run_example(
            RunExampleInput(
                tool_name=tool_name,
                image_path=pick,
                endpoint_url=demo_url or None,
            )
        )
        if demo_result.ran and (demo_result.result_preview or demo_result.result_image):
            preview_path = demo_result.result_preview or demo_result.result_image
            text += "✅ Demo completed!\n\n"
            images.append(preview_path)
            if demo_result.result_origin:
                files.append((demo_result.result_origin, "Download result"))
        else:
            note = demo_result.notes or "No output image returned"
            text += f"ℹ️ Demo ran but {note}"

        session.tool_calls.append(
            {
                "tool": "run_example",
                "tool_name": tool_name,
                "ran": demo_result.ran,
                "endpoint_url": demo_result.endpoint_url,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        log.exception("Demo execution failed")
        text += f"❌ Error: {e}"

    session.conversation_history.append(f"Assistant: {text}")
    return ChatTurnResult(
        status="tool_executed", text=text, images=images, files=files
    )


def _execute_registered_tool(
    session: Session, tool_name: str, params: Dict[str, Any]
) -> ChatTurnResult:
    """Execute a registered tool that gated on user approval."""
    tool_config = get_tool(tool_name)
    if not tool_config:
        text = f"❌ Error: Unknown tool '{tool_name}'"
        session.pending_tool_approval = None
        session.pending_tool_params = {}
        session.conversation_history.append(f"Assistant: {text}")
        return ChatTurnResult(status="error", text=text, error="unknown_tool")

    started = time.time()
    text = f"{tool_config.icon} Running {tool_config.display_name}...\n\n"
    images: List[str] = []
    files: List[tuple] = []
    try:
        # Backfill missing image path from last upload
        if "image_path" in params and not params["image_path"]:
            paths = session.last_asset_paths()
            if paths:
                params["image_path"] = paths[0]

        input_obj = tool_config.input_model(**params)
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
                images.append(preview_path)
            for dp in extract_downloads(result, tool_name):
                if os.path.exists(dp):
                    files.append((dp, f"Download {tool_config.display_name} result"))
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

    session.pending_tool_approval = None
    session.pending_tool_params = {}
    elapsed = time.time() - started
    log.info("Tool %s finished in %.2fs", tool_name, elapsed)
    session.conversation_history.append(f"Assistant: {text}")
    return ChatTurnResult(
        status="tool_executed", text=text, images=images, files=files
    )


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
