"""Gradio-specific application launcher.

All frontend-agnostic setup (logging, pipeline, tool registration, doc index)
lives in ``core.app_setup``.  This module only contains Gradio launch logic.
"""

from __future__ import annotations

import os
import socket
import logging

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)

log = logging.getLogger("chat_app")

# ---------------------------------------------------------------------------
# Re-export infrastructure helpers so existing callers (cli.py, etc.)
# that import from ui.app keep working.
# ---------------------------------------------------------------------------
from ai_agent.core.app_setup import (  # noqa: F401
    ensure_logging_initialized,
    ensure_tools_registered_once as _ensure_tools_registered_once,
    get_pipeline,
    get_doc_index,
    refresh_docs_from_index as refresh_ui_docs_from_index,
)

from ai_agent.ui.components import create_chat_interface


# ============================================================================
# Launch configuration  (Gradio-specific)
# ============================================================================
def _bind_host() -> str:
    """Determine bind host based on environment."""
    bind_host = os.getenv("BIND_HOST")
    if bind_host:
        return bind_host
    in_docker = os.path.exists("/.dockerenv")
    return "0.0.0.0" if in_docker else "127.0.0.1"


def launch():
    """Launch the Gradio chat interface."""
    ensure_logging_initialized()
    _ensure_tools_registered_once()

    doc_index = get_doc_index()

    host = _bind_host()
    preferred = int(os.getenv("PORT", "7860"))
    max_tries = int(os.getenv("PORT_TRIES", "10"))
    allow_fallback = str(os.getenv("ALLOW_PORT_FALLBACK", "1")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    ui = create_chat_interface(doc_index)

    def _is_port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return True
            except OSError:
                return False

    if allow_fallback:
        chosen_port = None
        for attempt in range(max_tries):
            candidate = preferred + attempt
            if _is_port_available(candidate):
                chosen_port = candidate
                break
        if chosen_port is not None:
            ui.queue(max_size=10).launch(
                server_name=host,
                server_port=chosen_port,
                inbrowser=False,
                show_error=True,
                share=bool(os.getenv("SHARE", False)),
            )
            if chosen_port != preferred:
                log.info(
                    "Launched on fallback port %d (preferred %d was busy)",
                    chosen_port,
                    preferred,
                )
            return

    last_err = None
    for attempt in range(max_tries if allow_fallback else 1):
        port = preferred + attempt
        try:
            ui.queue(max_size=10).launch(
                server_name=host,
                server_port=port,
                inbrowser=False,
                show_error=True,
                share=bool(os.getenv("SHARE", False)),
            )
            if attempt > 0:
                log.info(
                    "Launched on fallback port %d (preferred %d was busy)",
                    port,
                    preferred,
                )
            return
        except OSError as e:
            last_err = e
            busy = "Cannot find empty port" in str(e)
            if not busy or attempt == (max_tries - 1) or not allow_fallback:
                raise
            log.warning("Port %d busy; trying %d", port, port + 1)

    if last_err:
        raise last_err


if __name__ == "__main__":
    launch()
