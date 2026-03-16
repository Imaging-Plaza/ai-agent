from __future__ import annotations

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
