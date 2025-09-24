from __future__ import annotations
import re
from typing import List

# Matches any control tag we support
TAG_RE = re.compile(r"\[(?:REFINE|NO_RERANK|EXCLUDE:[^\]]*|EXCLUDED:[^\]]*)\]")
EXCLUDE_RE = re.compile(r"\[(?:EXCLUDE|EXCLUDED):([^\]]+)\]")

def strip_tags(text: str) -> str:
    """Remove all control tags from text."""
    if not text:
        return text
    return TAG_RE.sub("", text).strip()

def parse_exclusions(text: str) -> List[str]:
    """Return names from [EXCLUDE:a|b] or [EXCLUDED:a|b] (empty if none)."""
    if not text:
        return []
    m = EXCLUDE_RE.search(text)
    if not m:
        return []
    parts = [p.strip() for p in m.group(1).split("|")]
    return [p for p in parts if p]

def has_no_rerank(text: str) -> bool:
    return "[NO_RERANK]" in (text or "")

def has_refine(text: str) -> bool:
    return "[REFINE]" in (text or "")

# --- refinement intent helpers ---------------------------------------------

_REFINE_KEYWORDS = (
    "alternative",
    "alternatives",
    "another option",
    "another tool",
    "other option",
    "other options",
    "something else",
    "different tool",
    "different software",
    "show other",
    "find other",
    "find alternatives",
    "more options",
    "more tools",
)

def is_refine_intent(text: str) -> bool:
    t = (text or "").lower()
    if "[refine]" in t:
        return True
    return any(k in t for k in _REFINE_KEYWORDS)

def strip_refine_keywords(text: str) -> str:
    t = (text or "")
    low = t.lower()
    for k in _REFINE_KEYWORDS:
        if k in low:
            # crude but effective: remove the matching span
            idx = low.find(k)
            t = t[:idx] + t[idx + len(k):]
            low = t.lower()
    return t.strip()
