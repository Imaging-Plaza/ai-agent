# src/ai_agent/cli.py
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dotenv import load_dotenv
import logging

load_dotenv()
log = logging.getLogger("ai_agent.cli")

from ai_agent.catalog.sync import sync_once


def _ui_funcs():
    # Lazy import avoids loading agent/model modules for non-UI commands.
    from ai_agent.ui.app import (
        get_pipeline,
        refresh_ui_docs_from_index,
        launch,
        ensure_logging_initialized,
    )

    return get_pipeline, refresh_ui_docs_from_index, launch, ensure_logging_initialized


# --------------------------- catalog background refresher ---------------------------
def _background_refresh():
    """If SYNC_EVERY_HOURS > 0, refresh in the background while UI runs."""
    hours = float(os.getenv("SYNC_EVERY_HOURS", "0") or 0)

    if hours <= 0:
        log.info("[auto-refresh] disabled")
        return

    def _loop():
        # Startup already performs one sync in run_chat(); wait one full interval
        # before the first background refresh to avoid duplicate work.
        interval_s = max(60.0, hours * 3600.0)
        time.sleep(interval_s)

        while True:
            try:
                res = sync_once()
                log.info(
                    "[auto-refresh] %s → %s",
                    res.get("count", "?"),
                    res.get("jsonl_path"),
                )

                get_pipeline, refresh_ui_docs_from_index, _, _ = _ui_funcs()
                pipe = get_pipeline()

                if res.get("changed"):
                    ok = pipe.reload_index()
                    if ok:
                        log.info("[auto-refresh] reloaded FAISS index")
                        refresh_ui_docs_from_index()
                    else:
                        log.warning(
                            "[auto-refresh] reload failed; serving previous index"
                        )
                else:
                    log.info("[auto-refresh] catalog unchanged; FAISS not touched")
            except Exception:
                log.exception("[auto-refresh] error")
            try:
                time.sleep(interval_s)
            except Exception:
                time.sleep(3600.0)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# --------------------------- custom tasks ---------------------------
def run_chat():
    """Launch the chat-based UI."""
    try:
        _, _, _, ensure_logging_initialized = _ui_funcs()
        ensure_logging_initialized()

        res = sync_once()
        log.info("[startup-sync] %s → %s", res.get("count", "?"), res.get("jsonl_path"))

        get_pipeline, refresh_ui_docs_from_index, launch, _ = _ui_funcs()

        # Initialize pipeline
        pipe = get_pipeline()

        if res.get("changed"):
            ok = pipe.reload_index()
            if ok:
                log.info("[startup-refresh] reloaded FAISS index")
                refresh_ui_docs_from_index()
            else:
                log.warning("[startup-refresh] reload failed; serving previous index")
        else:
            log.info(
                "[startup-refresh] catalog unchanged; keeping existing FAISS index"
            )
    except Exception:
        log.exception("[startup-sync] failed")

    _background_refresh()

    try:
        _, _, launch, _ = _ui_funcs()
        launch()
    except Exception:
        log.exception("[chat-launch] failed")
        raise


def run_sync():
    try:
        r = sync_once()
        log.info("[sync] %s → %s", r.get("count", "?"), r.get("jsonl_path"))
    except Exception:
        log.exception("[sync] failed")
        raise


# --------------------------- main entry ---------------------------
def main():
    p = argparse.ArgumentParser(description="AI Agent CLI")
    p.add_argument(
        "mode",
        choices=["chat", "sync"],
        help="'chat' launches the chat UI; 'sync' runs one catalog refresh.",
    )
    args = p.parse_args()

    if args.mode == "chat":
        run_chat()
    elif args.mode == "sync":
        run_sync()
    else:
        p.print_help()
        sys.exit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
