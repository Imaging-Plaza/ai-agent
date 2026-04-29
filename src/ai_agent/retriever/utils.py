from __future__ import annotations

from typing import Optional


def _resolve_local_device(device: Optional[str]) -> str:
    """Resolve device for local sentence-transformers models.

    Priority: explicit device -> cuda -> mps -> cpu.
    """
    if device and str(device).strip().lower() != "auto":
        return str(device).strip()

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"