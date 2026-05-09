# utils/cache_db.py
"""Thread-safe SQLite-backed cache with TTL and LRU eviction.

Replaces the in-memory dict/OrderedDict caches used by preview, image-metadata,
and repo-info modules.  A single database file holds all namespaces in one table
so there is exactly one connection to manage.

Usage::

    from ai_agent.utils.cache_db import get_cache_db

    db = get_cache_db()
    db.set("meta", key, value, max_entries=128)
    hit = db.get("meta", key)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("cache_db")


def _default_db_path() -> str:
    """Return a per-user cache DB path so different OS users cannot share data.

    On POSIX systems the current UID is embedded in the filename, ensuring
    each user gets an isolated file under the system temp directory.  The
    caller can override the path entirely via ``CACHE_DB_PATH``.
    """
    env = os.getenv("CACHE_DB_PATH")
    if env:
        return env
    uid_suffix = ""
    try:
        uid_suffix = f"_{os.getuid()}"
    except AttributeError:
        pass  # Windows — no getuid(); fall back to a single shared filename
    return str(Path(tempfile.gettempdir()) / f"ai_agent_cache{uid_suffix}.db")


_DEFAULT_DB_PATH = _default_db_path()


class CacheDB:
    """Thread-safe SQLite-backed key/value cache.

    * **Namespaces** isolate unrelated caches inside one DB.
    * **TTL** – entries expire after *ttl_seconds* (wallclock).
      ``ttl_seconds <= 0`` means the entry never expires.
    * **Capacity** – when *max_entries > 0*, the oldest-accessed rows in
      that namespace are evicted after each write.
    * All values are stored as plain text; callers handle serialisation.
    """

    @staticmethod
    def _ensure_private_db_file(path: str) -> None:
        """Ensure an on-disk SQLite DB file is owner-readable/writable only.

        This is a best-effort hardening step for local cache data. For new
        files, create the file first so we can set restrictive permissions
        before SQLite opens it. For existing files, tighten permissions.
        """
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            os.chmod(path, 0o600)
        else:
            os.close(fd)
            os.chmod(path, 0o600)

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = str(db_path or _DEFAULT_DB_PATH)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._ensure_private_db_file(path)

        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()

        # Restrict the DB file to the owner only so other OS users cannot
        # read cached data.  The chmod is best-effort (no-op on Windows).
        if path != ":memory:":
            try:
                os.chmod(path, 0o600)
            except Exception:
                log.warning("Could not set owner-only permissions on cache DB: %s", path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache (
                namespace   TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                expires_at  REAL,          -- NULL → never expires
                accessed_at REAL NOT NULL,
                PRIMARY KEY (namespace, key)
            );
            CREATE INDEX IF NOT EXISTS idx_cache_ns_expires
                ON cache (namespace, expires_at);
            CREATE INDEX IF NOT EXISTS idx_cache_ns_accessed
                ON cache (namespace, accessed_at);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, namespace: str, key: str, *, track_lru: bool = True
    ) -> Optional[str]:
        """Return the cached value or *None* if missing / expired.

        *track_lru* (default ``True``) updates *accessed_at* on a hit so that
        LRU eviction in :meth:`set` works correctly.  Pass ``False`` when the
        namespace has no capacity limit and you want to avoid the extra write.
        """
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM cache"
                " WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at is not None and expires_at <= now:
                self._conn.execute(
                    "DELETE FROM cache WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
                self._conn.commit()
                return None
            if track_lru:
                # Touch accessed_at so LRU eviction in set() stays accurate.
                self._conn.execute(
                    "UPDATE cache SET accessed_at = ? WHERE namespace = ? AND key = ?",
                    (now, namespace, key),
                )
                self._conn.commit()
            return value

    def set(
        self,
        namespace: str,
        key: str,
        value: str,
        *,
        ttl_seconds: float = 0,
        max_entries: int = 0,
    ) -> None:
        """Store *value* under *(namespace, key)*.

        *ttl_seconds ≤ 0* → no expiration.
        *max_entries ≤ 0*  → no capacity limit.
        """
        now = time.time()
        expires_at = (now + ttl_seconds) if ttl_seconds > 0 else None
        with self._lock:
            # Lazy sweep of expired rows in this namespace
            self._conn.execute(
                "DELETE FROM cache"
                " WHERE namespace = ? AND expires_at IS NOT NULL AND expires_at <= ?",
                (namespace, now),
            )
            self._conn.execute(
                """
                INSERT INTO cache (namespace, key, value, expires_at, accessed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (namespace, key)
                DO UPDATE SET value       = excluded.value,
                              expires_at  = excluded.expires_at,
                              accessed_at = excluded.accessed_at
                """,
                (namespace, key, value, expires_at, now),
            )
            # Enforce capacity by evicting least-recently-accessed rows
            if max_entries > 0:
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM cache WHERE namespace = ?",
                    (namespace,),
                ).fetchone()[0]
                overage = count - max_entries
                if overage > 0:
                    self._conn.execute(
                        """
                        DELETE FROM cache
                        WHERE rowid IN (
                            SELECT rowid FROM cache
                            WHERE namespace = ?
                            ORDER BY accessed_at ASC
                            LIMIT ?
                        )
                        """,
                        (namespace, overage),
                    )
            self._conn.commit()

    def delete(self, namespace: str, key: str) -> None:
        """Remove a single entry."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            self._conn.commit()

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear entries.  *None* → all namespaces."""
        with self._lock:
            if namespace is not None:
                self._conn.execute(
                    "DELETE FROM cache WHERE namespace = ?", (namespace,)
                )
            else:
                self._conn.execute("DELETE FROM cache")
            self._conn.commit()

    def sweep_expired(self) -> int:
        """Delete all expired rows across every namespace.

        Returns the number of rows removed.  Designed to be called from a
        background thread or an atexit hook without touching private state.
        """
        now = time.time()
        with self._lock:
            deleted = self._conn.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            ).rowcount
            self._conn.commit()
        return deleted

    def vacuum_and_close(self) -> None:
        """Sweep expired rows, VACUUM the database, then close the connection.

        VACUUM must run outside any open transaction; this method handles the
        ``isolation_level`` toggling safely with a ``finally`` guard.
        """
        self.sweep_expired()
        with self._lock:
            previous_isolation_level = self._conn.isolation_level
            try:
                self._conn.isolation_level = None
                self._conn.execute("VACUUM")
            finally:
                self._conn.isolation_level = previous_isolation_level
        self.close()

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_db: Optional[CacheDB] = None
_db_lock = threading.Lock()


def get_cache_db() -> CacheDB:
    """Return (or create) the process-wide :class:`CacheDB` singleton."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = CacheDB()
                log.info("Opened cache DB at %s", _DEFAULT_DB_PATH)
    return _db


def get_cache_db_or_none() -> Optional[CacheDB]:
    """Return the singleton if it has already been initialised, else ``None``.

    Unlike :func:`get_cache_db` this never creates the database — useful for
    background housekeeping that should be a no-op before the first real cache
    access.
    """
    return _db


def reset_cache_db(db: Optional[CacheDB] = None) -> None:
    """Replace the singleton — mainly useful for tests.

    If *db* is ``None`` a fresh in-memory database is **not** created
    automatically; the next call to :func:`get_cache_db` will open the
    default path again.
    """
    global _db
    with _db_lock:
        if _db is not None:
            _db.close()
        _db = db
