"""Chat router with SSE streaming.

Event protocol (named events; payloads are JSON):

  ``session``        ``{"session_id": "..."}``                            once, first
  ``user_message``   ``{"text": "...", "asset_ids": [...]}``              echo
  ``status``         ``{"phase": "thinking" | "agent_running" | ...}``    occasional
  ``text``           ``{"content": "..."}``                                main reply
  ``recommendation`` ``{"item": {rank, name, accuracy, why, ...}}``       per card
  ``tool_trace``     ``{"trace": {...}}``                                  per call
  ``pending_action`` ``{"action": {type, tool_name, ...}}``                if pause
  ``clarification``  ``{"question": "...", "options": [...]}``             if needed
  ``images``         ``{"paths": [...]}``                                  tool-exec
  ``files``          ``{"items": [{path, label}, ...]}``                   tool-exec
  ``usage``          ``{"total": n, "input": n, "output": n}``             once
  ``error``          ``{"message": "...", "code": "..."}``                 fatal
  ``done``           ``{}``                                                terminal

The underlying agent (``services.chat.process_turn``) is still synchronous;
this router translates the final ``ChatTurnResult`` into a sequence of
events. As soon as pydantic-ai's ``run_stream`` is wired in, the same
endpoint will emit ``text`` deltas live.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from ai_agent.api.deps import get_doc_index, require_auth
from ai_agent.api.schemas import ChatStartBody
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.services.chat import (
    ChatRequest,
    ChatTurnResult,
    approve_pending,
    decline_pending,
    process_turn,
)
from ai_agent.services.sessions import Session, get_session_store

log = logging.getLogger("api.routers.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(require_auth)])


def _event(name: str, data: Any) -> Dict[str, Any]:
    return {"event": name, "data": json.dumps(data, default=str)}


def _stream_result(
    session: Session, result: ChatTurnResult
) -> AsyncIterator[Dict[str, Any]]:
    """Translate a finished ChatTurnResult into a sequence of SSE events."""

    async def gen() -> AsyncIterator[Dict[str, Any]]:
        # Always lead with the session id so the client can persist it.
        yield _event("session", {"session_id": session.session_id})

        if result.text:
            yield _event("text", {"content": result.text})

        for rec in result.recommendations:
            yield _event(
                "recommendation",
                {
                    "rank": rec.rank,
                    "name": rec.name,
                    "accuracy": rec.accuracy,
                    "why": rec.why,
                    "doc": rec.doc,
                    "demo_url": rec.demo_url,
                },
            )

        for trace in result.tool_traces:
            yield _event("tool_trace", {"trace": trace})

        if result.clarification:
            yield _event(
                "clarification",
                {
                    "question": result.clarification.question,
                    "context": result.clarification.context,
                    "options": result.clarification.options,
                },
            )

        if result.pending_action:
            pa = result.pending_action
            yield _event(
                "pending_action",
                {
                    "type": pa.type,
                    "tool_name": pa.tool_name,
                    "display_name": pa.display_name,
                    "icon": pa.icon,
                    "image_name": pa.image_name,
                    "demo_url": pa.demo_url,
                    "prompt": pa.prompt,
                },
            )

        if result.images:
            yield _event("images", {"paths": result.images})
        if result.files:
            yield _event(
                "files",
                {"items": [{"path": p, "label": label} for p, label in result.files]},
            )

        if result.usage:
            yield _event("usage", result.usage)

        if result.status == "error" and result.error:
            yield _event("error", {"message": result.text, "code": result.error})

        yield _event("done", {"status": result.status})

    return gen()


def _resolve_or_error(session_id: str) -> Session:
    session = get_session_store().get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found"
        )
    return session


@router.post("")
async def start_chat(
    body: ChatStartBody,
    doc_index: Dict[str, SoftwareDoc] = Depends(get_doc_index),
):
    store = get_session_store()
    session = store.get_or_create(body.session_id)
    request = ChatRequest(
        message=body.message,
        asset_ids=list(body.asset_ids),
        model=body.model,
        top_k=body.top_k,
        num_choices=body.num_choices,
    )
    # Run the (currently synchronous) agent in a thread so the event loop is
    # free while it executes. When run_stream lands we can ditch this.
    result = await asyncio.to_thread(process_turn, session, request, doc_index)
    return EventSourceResponse(_stream_result(session, result))


@router.post("/{session_id}/approve")
async def approve(
    session_id: str,
    doc_index: Dict[str, SoftwareDoc] = Depends(get_doc_index),
):
    session = _resolve_or_error(session_id)
    result = await asyncio.to_thread(approve_pending, session)
    return EventSourceResponse(_stream_result(session, result))


@router.post("/{session_id}/decline")
async def decline(session_id: str):
    session = _resolve_or_error(session_id)
    result = decline_pending(session)
    return EventSourceResponse(_stream_result(session, result))


@router.post("/{session_id}/confirm-demo")
async def confirm_demo(
    session_id: str,
    doc_index: Dict[str, SoftwareDoc] = Depends(get_doc_index),
):
    """Resume a turn that asked the user to confirm running a demo URL.

    Sends an affirmative reply through the normal chat pipeline so the
    existing pending-demo logic fires.
    """
    session = _resolve_or_error(session_id)
    request = ChatRequest(message="yes")
    result = await asyncio.to_thread(process_turn, session, request, doc_index)
    return EventSourceResponse(_stream_result(session, result))
