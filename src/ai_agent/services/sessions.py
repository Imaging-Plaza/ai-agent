"""In-process session store with TTL.

Sessions are keyed by an opaque ``session_id`` and hold the per-conversation
state that used to live inside Gradio's ``gr.State``: conversation history,
uploaded assets, the agent's banlist, pending tool / demo approvals, and the
running list of tool-call records.

A single ``SessionStore`` instance is shared between the Gradio adapter and
the FastAPI routers; both call ``get_or_create``. TTL eviction keeps memory
bounded without a background sweeper — expired entries are pruned on touch.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("services.sessions")

DEFAULT_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 6)))  # 6h


@dataclass
class Asset:
    """A single uploaded multimodal asset (image, DICOM, NIfTI, …).

    ``path`` is the original file on disk (kept for tool execution and the
    agent's metadata extraction). ``preview_path`` is the PNG/GIF preview fed
    to the VLM; ``metadata_text`` is the compact human-readable summary the
    agent uses as hidden context.
    """

    asset_id: str
    path: str
    preview_path: Optional[str] = None
    metadata_text: Optional[str] = None
    original_format: Optional[str] = None
    display_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Conversation
    conversation_history: List[str] = field(default_factory=list)
    """Free-text running transcript used to prompt the agent."""

    # Multimodal
    assets: Dict[str, Asset] = field(default_factory=dict)
    last_asset_ids: List[str] = field(default_factory=list)
    """The asset_ids attached to the most recent user turn — reused when a
    follow-up turn omits attachments."""

    # Agent retrieval state
    banlist: set = field(default_factory=set)
    last_choices: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    # Pending actions (require a follow-up POST to resume)
    pending_demo_tool: Optional[str] = None
    pending_demo_url: Optional[str] = None
    pending_tool_approval: Optional[str] = None
    pending_tool_params: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_seen = time.time()

    def last_asset_paths(self) -> List[str]:
        return [self.assets[a].path for a in self.last_asset_ids if a in self.assets]

    def last_preview(self) -> Optional[Asset]:
        for aid in reversed(self.last_asset_ids):
            asset = self.assets.get(aid)
            if asset and asset.preview_path:
                return asset
        return None


class SessionStore:
    """Thread-safe in-process session store with TTL eviction-on-touch."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def _evict_expired(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        expired = [
            sid for sid, s in self._sessions.items() if now - s.last_seen > self._ttl
        ]
        for sid in expired:
            log.info("Evicting expired session %s", sid)
            self._sessions.pop(sid, None)

    def create(self) -> Session:
        sid = str(uuid.uuid4())
        with self._lock:
            self._evict_expired()
            session = Session(session_id=sid)
            self._sessions[sid] = session
        log.info("Created session %s", sid)
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is not None:
                session.touch()
            return session

    def get_or_create(self, session_id: Optional[str]) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        return self.create()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


_store: Optional[SessionStore] = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """Process-wide singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SessionStore()
    return _store


__all__ = ["Asset", "Session", "SessionStore", "get_session_store"]
