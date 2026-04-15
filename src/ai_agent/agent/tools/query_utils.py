from __future__ import annotations

import re
from typing import List

FORMAT_TOKEN_MAP = {
    "tif": "TIFF",
    "tiff": "TIFF",
    "nii": "NIfTI",
    "nii.gz": "NIfTI",
    "dcm": "DICOM",
    "dicom": "DICOM",
    "nrrd": "NRRD",
    "png": "PNG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
}

_REPO_DRIFT_TERMS = {
    "github",
    "repository",
    "repo",
    "official",
    "readme",
    "docs",
    "documentation",
    "source",
    "sourcecode",
}

_LOW_SIGNAL_TERMS = _REPO_DRIFT_TERMS | {
    "tool",
    "tools",
    "project",
    "framework",
}


def _tokenize_query(query: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9_+-]+", (query or "").lower()) if t]


def normalize_formats(formats: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for ext in formats:
        norm = (ext or "").strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def append_format_tokens(query: str, formats: List[str]) -> str:
    fmt_tokens: List[str] = []
    for ext in normalize_formats(formats):
        canon = FORMAT_TOKEN_MAP.get(ext, ext.upper())
        if canon not in fmt_tokens:
            fmt_tokens.append(canon)

    if not fmt_tokens:
        return query.strip()
    return (query.strip() + " " + " ".join(f"format:{t}" for t in fmt_tokens)).strip()


def strip_legacy_original_formats_line(query: str) -> tuple[str, List[str]]:
    """Parse and remove legacy OriginalFormats: line from query text."""
    original_formats: List[str] = []
    clean_lines = []

    for line in (query or "").splitlines():
        if line.lower().startswith("originalformats:"):
            parts = line.split(":", 1)[1].strip().split()
            original_formats.extend(parts)
            continue
        clean_lines.append(line)

    base_query = " ".join(ln.strip() for ln in clean_lines if ln.strip())
    return base_query, normalize_formats(original_formats)


def sanitize_retrieval_query(
    query: str,
    known_tool_names: List[str] | None = None,
    fallback_query: str | None = None,
) -> str:
    """
    Sanitize LLM-generated retrieval queries by removing repository drift terms.

    If the query collapses into a tool-name-only or low-signal query, fallback
    to the previous task-centric query when provided.
    """
    raw = (query or "").strip()
    if not raw:
        return (fallback_query or "").strip()

    # Remove URLs and punctuation-heavy fragments.
    s = re.sub(r"https?://\S+", " ", raw, flags=re.IGNORECASE)
    s = re.sub(r"www\.\S+", " ", s, flags=re.IGNORECASE)

    tokens = _tokenize_query(s)
    if not tokens:
        return (fallback_query or raw).strip()

    filtered = [t for t in tokens if t not in _REPO_DRIFT_TERMS]
    if not filtered:
        return (fallback_query or raw).strip()

    # Detect tool-name-only drift (e.g., "dhsegment official github repository").
    if known_tool_names:
        token_set = set(filtered)
        for nm in known_tool_names:
            nm_tokens = set(_tokenize_query(nm))
            if not nm_tokens:
                continue
            if token_set.issubset(nm_tokens) and len(token_set) <= 3:
                return (fallback_query or " ".join(filtered)).strip()

    low_signal = all(t in _LOW_SIGNAL_TERMS for t in filtered)
    if low_signal:
        return (fallback_query or " ".join(filtered)).strip()

    # If it became too short and a previous task query exists, prefer that.
    if len(filtered) <= 2 and fallback_query:
        return fallback_query.strip()

    return " ".join(filtered).strip()
