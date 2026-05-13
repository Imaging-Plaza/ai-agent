"""Gradio adapter on top of the transport-agnostic chat services.

Keeps the historic ``respond`` / ``execute_tool_with_approval`` signatures so
``ui.components`` and ``ui.app`` don't need to change, while delegating all
real work to ``ai_agent.services``.

Per-conversation state now lives in a :class:`Session` managed by the global
:class:`SessionStore`. The Gradio ``gr.State`` only carries the session id
(plus a mirror of a few fields the visualizations panel reads directly).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.services.chat import (
    ChatRequest,
    ChatTurnResult,
    approve_pending,
    process_turn,
)
from ai_agent.services.sessions import Session, get_session_store
from ai_agent.utils.utils import _coerce_files_to_paths

from .formatters import format_tool_card
from .state import ChatMessage, ChatState

log = logging.getLogger("chat_handlers")


# ---------------------------------------------------------------------------
# State bridge: gr.State <-> Session
# ---------------------------------------------------------------------------
def _session_from_state(state_dict: dict) -> Tuple[Session, dict]:
    """Pull or create a Session for this Gradio session.

    Reads ``state_dict['session_id']`` if present; falls back to creating a
    new session. The dict is updated in place so the caller can return it
    back to Gradio.
    """
    state_dict = dict(state_dict or {})
    store = get_session_store()
    sid = state_dict.get("session_id")
    if sid:
        session = store.get(sid)
        if session is None:
            session = store.create()
            state_dict["session_id"] = session.session_id
    else:
        session = store.create()
        state_dict["session_id"] = session.session_id
    return session, state_dict


def _mirror_session(session: Session, state_dict: dict) -> ChatState:
    """Copy the subset of Session fields the visualizations panel reads."""
    state_dict["session_id"] = session.session_id
    state_dict["tool_calls"] = list(session.tool_calls)
    state_dict["banlist"] = list(session.banlist)
    state_dict["last_choices"] = dict(session.last_choices)
    state_dict["conversation_history"] = list(session.conversation_history)
    state_dict["pending_demo_tool"] = session.pending_demo_tool
    state_dict["pending_demo_url"] = session.pending_demo_url
    state_dict["pending_tool_approval"] = session.pending_tool_approval
    state_dict["pending_tool_params"] = dict(session.pending_tool_params)
    state_dict["last_files"] = session.last_asset_paths()
    state_dict["last_preview_path"] = (
        session.last_preview().preview_path if session.last_preview() else None
    )
    state_dict["last_image_meta"] = (
        session.last_preview().metadata_text if session.last_preview() else None
    )

    # Construct a legacy ChatState for return-type compatibility
    return ChatState.from_dict(state_dict)


def _result_to_message(
    result: ChatTurnResult, doc_index: Dict[str, SoftwareDoc]
) -> ChatMessage:
    """Render a service result into a Gradio ``ChatMessage``."""
    reply = ChatMessage()

    parts: List[str] = []
    if result.text:
        parts.append(result.text)

    if result.recommendations:
        parts.append("---\n")
        for rec in result.recommendations:
            if rec.doc:
                doc = SoftwareDoc.model_validate(rec.doc)
                parts.append(format_tool_card(doc, rec.accuracy, rec.why, rec.rank))
            else:
                # Fallback when doc isn't in the index
                parts.append(
                    f"**{rec.rank}. {rec.name}** — {rec.accuracy:.1f}%\n\n_{rec.why}_\n"
                )

    if result.pending_action:
        pa = result.pending_action
        if pa.type == "tool_approval":
            parts.append(f"\n🚀 **Ready to run {pa.display_name or pa.tool_name}?**\n")
            if pa.image_name:
                parts.append(f"📁 **Image:** {pa.image_name}\n")
            if pa.demo_url:
                parts.append(f"🔗 **Endpoint:** {pa.demo_url}\n")
            parts.append(
                f"_Press the **'{pa.icon or '🚀'} Run Tool'** button below, "
                "or ask about other tools in the chat instead._"
            )
        elif pa.type == "demo_confirm":
            parts.append(
                f"\n💡 **Would you like me to run the demo for {pa.tool_name}?**\n"
            )
            if pa.demo_url:
                parts.append(f"🔗 Demo: {pa.demo_url}\n")
            parts.append(
                "_Press the **'🚀 Run Demo'** button to run the demo, or "
                "continue with another request._"
            )

    reply.text = "\n".join(parts)
    reply.tool_traces = list(result.tool_traces)
    if result.usage:
        reply.stats = {
            "tokens": {
                "total": result.usage.get("total", 0),
                "input": result.usage.get("input", 0),
                "output": result.usage.get("output", 0),
            }
        }
    reply.images.extend(result.images)
    reply.files.extend(result.files)
    return reply


# ---------------------------------------------------------------------------
# Public functions (signatures preserved for Gradio app)
# ---------------------------------------------------------------------------
def respond(
    message: str,
    files: List[Any],
    state_dict: dict,
    doc_index: Dict[str, SoftwareDoc],
    model: str | None = None,
    top_k: int | None = None,
    num_choices: int | None = None,
) -> Tuple[ChatMessage, ChatState]:
    """Run one chat turn; ChatState is mirrored from the underlying Session."""
    session, state_dict = _session_from_state(state_dict)
    request = ChatRequest(
        message=message or "",
        new_file_paths=_coerce_files_to_paths(files) or [],
        model=model,
        top_k=top_k,
        num_choices=num_choices,
    )
    result = process_turn(session, request, doc_index)
    reply = _result_to_message(result, doc_index)
    state = _mirror_session(session, state_dict)
    return reply, state


def execute_tool_with_approval(
    tool_name: str,
    tool_params: Dict[str, Any],
    state: ChatState,
) -> Tuple[ChatMessage, ChatState]:
    """Resume a turn that left a tool waiting for user approval.

    ``tool_name`` / ``tool_params`` are kept in the signature for backwards
    compatibility but are sourced from the session — they always match what
    was captured when the agent recommended the tool.
    """
    state_dict = state.to_dict() if isinstance(state, ChatState) else dict(state or {})
    session, state_dict = _session_from_state(state_dict)

    # Override params if the caller provided fresh ones (e.g., image_path
    # backfilled from a fresh upload). This keeps the historic behaviour.
    if tool_params:
        merged = dict(session.pending_tool_params)
        for k, v in tool_params.items():
            if v not in (None, ""):
                merged[k] = v
        session.pending_tool_params = merged
        # Also align name so approve_pending picks the requested tool
        session.pending_tool_approval = tool_name

    result = approve_pending(session)
    reply = _result_to_message(result, {})
    state_out = _mirror_session(session, state_dict)
    return reply, state_out


__all__ = ["respond", "execute_tool_with_approval"]
