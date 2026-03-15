"""
Centralized temporary file management with automatic cleanup on shutdown.
"""

from __future__ import annotations

import os
import logging
import atexit
import threading
from typing import Optional

log = logging.getLogger("utils.temp_file_manager")

# Global registry of all temporary files across all tools
_temp_files: list[str] = []
_cleanup_registered = False
_lock = threading.Lock()


def register_temp_file(path: Optional[str]) -> Optional[str]:
    """
    Register a temporary file for cleanup on shutdown.
    Thread-safe for multi-user Gradio deployments.

    Args:
        path: Path to temporary file

    Returns:
        The same path (pass-through for convenience)
    """
    global _cleanup_registered

    if not path:
        return path

    with _lock:
        if path not in _temp_files:
            _temp_files.append(path)

            # Register cleanup on first use
            if not _cleanup_registered:
                atexit.register(cleanup_temp_files)
                _cleanup_registered = True

    return path


def cleanup_temp_files() -> None:
    """Clean up all registered temporary files. Thread-safe."""
    with _lock:
        if not _temp_files:
            return

        log.info(f"Cleaning up {len(_temp_files)} temporary file(s)")

        for path in _temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    log.debug(f"Cleaned up temporary file: {path}")
            except Exception as e:
                log.warning(f"Failed to clean up {path}: {e}")

        _temp_files.clear()


def get_temp_file_count() -> int:
    """Get the number of registered temporary files."""
    with _lock:
        return len(_temp_files)
