"""
Centralized temporary file management with automatic cleanup on shutdown.
"""
from __future__ import annotations

import os
import logging
import atexit
from typing import Optional

log = logging.getLogger("utils.temp_file_manager")

# Global registry of all temporary files across all tools
_temp_files: list[str] = []
_cleanup_registered = False


def register_temp_file(path: Optional[str]) -> Optional[str]:
    """
    Register a temporary file for cleanup on shutdown.
    
    Args:
        path: Path to temporary file
        
    Returns:
        The same path (pass-through for convenience)
    """
    global _cleanup_registered
    
    if path and path not in _temp_files:
        _temp_files.append(path)
        
        # Register cleanup on first use
        if not _cleanup_registered:
            atexit.register(cleanup_temp_files)
            _cleanup_registered = True
    
    return path


def cleanup_temp_files() -> None:
    """Clean up all registered temporary files."""
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
    return len(_temp_files)
