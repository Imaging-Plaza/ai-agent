# utils/shutdown.py
"""Periodic cleanup and shutdown hooks.

Two mechanisms keep the app tidy at runtime and on exit:

**Background cleanup thread** (started by :func:`register`):
- Sweeps expired cache rows every ``CLEANUP_INTERVAL_SECONDS`` (default 3600).
- Purges old log files on the same interval (files older than
  ``LOG_RETENTION_DAYS`` days, default 7).

**atexit hook** (also registered by :func:`register`):
- Runs a final cache sweep, then VACUUM and closes the connection cleanly
  (triggers a WAL checkpoint).  Handles the case where the process exits
  before the next background interval fires.

Call :func:`register` once at startup (see ``cli.py``).
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("ai_agent.shutdown")

LOG_RETENTION_DAYS: int = int(os.getenv("LOG_RETENTION_DAYS", "7"))
CLEANUP_INTERVAL_SECONDS: int = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "7200"))


# ---------------------------------------------------------------------------
# Cache DB helpers
# ---------------------------------------------------------------------------

def _sweep_cache_db() -> None:
    """Delete expired rows from the cache DB (lightweight, runs periodically)."""
    from ai_agent.utils.cache_db import _db  # noqa: PLC0415

    if _db is None:
        return
    try:
        now = time.time()
        with _db._lock:
            deleted = _db._conn.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            ).rowcount
            _db._conn.commit()
        if deleted:
            log.debug("Cache sweep: removed %d expired row(s).", deleted)
    except Exception:
        log.exception("Periodic cache sweep failed.")


def _vacuum_and_close_cache_db() -> None:
    """Final shutdown: VACUUM and close the cache DB (runs via atexit)."""
    from ai_agent.utils.cache_db import _db, get_cache_db  # noqa: PLC0415

    if _db is None:
        return
    try:
        db = get_cache_db()
        _sweep_cache_db()
        with db._lock:
            # VACUUM must run outside any transaction.
            db._conn.isolation_level = None
            db._conn.execute("VACUUM")
            db._conn.isolation_level = ""
        log.info("Cache DB shutdown: VACUUM complete.")
        db.close()
    except Exception:
        log.exception("Cache DB shutdown cleanup failed.")


# ---------------------------------------------------------------------------
# Log file rotation helper
# ---------------------------------------------------------------------------

def _purge_old_logs() -> None:
    """Delete log files older than LOG_RETENTION_DAYS inside LOG_DIR.

    Age is determined by the date embedded in the filename (``app_YYYYMMDD``),
    which is reliable even when the file's mtime has been reset (e.g. the file
    was copied or recreated).  Falls back to mtime for files whose name does
    not contain a parseable date.
    """
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    if not log_dir.is_dir():
        return

    cutoff = time.time() - LOG_RETENTION_DAYS * 86_400
    _date_re = re.compile(r"app_(\d{8})")
    removed = 0
    errors = 0
    for entry in log_dir.iterdir():
        if not entry.is_file():
            continue
        if not (entry.name.startswith("app_") and ".log" in entry.name):
            continue
        try:
            # Prefer the date in the filename over mtime.
            m = _date_re.search(entry.name)
            if m:
                file_ts = datetime.strptime(m.group(1), "%Y%m%d").replace(
                    tzinfo=timezone.utc
                ).timestamp()
            else:
                file_ts = entry.stat().st_mtime

            if file_ts < cutoff:
                entry.unlink()
                removed += 1
        except Exception:
            log.exception("Failed to delete old log file: %s", entry)
            errors += 1

    if removed or errors:
        log.info(
            "Log cleanup: removed %d file(s) older than %d day(s)%s.",
            removed,
            LOG_RETENTION_DAYS,
            f", {errors} error(s)" if errors else "",
        )


# ---------------------------------------------------------------------------
# Background cleanup thread
# ---------------------------------------------------------------------------

def _cleanup_loop(interval: int, stop_event: threading.Event) -> None:
    """Run periodic sweeps until *stop_event* is set or the process exits.

    The first sweep runs immediately on startup so stale data is removed
    without waiting for the first interval to elapse.
    """
    while True:
        _sweep_cache_db()
        _purge_old_logs()
        if stop_event.wait(timeout=interval):
            break


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_stop_event: threading.Event | None = None


def register() -> None:
    """Start the background cleanup thread and register the atexit hook.

    Safe to call multiple times (idempotent).
    """
    global _stop_event

    # Stop any previously running background thread before restarting.
    if _stop_event is not None:
        _stop_event.set()

    _stop_event = threading.Event()
    thread = threading.Thread(
        target=_cleanup_loop,
        args=(CLEANUP_INTERVAL_SECONDS, _stop_event),
        name="cache-log-cleanup",
        daemon=True,  # won't block process exit
    )
    thread.start()
    log.info(
        "Background cleanup started (interval: %ds, log retention: %dd).",
        CLEANUP_INTERVAL_SECONDS,
        LOG_RETENTION_DAYS,
    )

    atexit.unregister(_vacuum_and_close_cache_db)
    atexit.register(_vacuum_and_close_cache_db)
