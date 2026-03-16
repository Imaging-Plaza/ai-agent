from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ai_agent.agent.tools.query_utils import (
    append_format_tokens,
    normalize_formats,
    strip_legacy_original_formats_line,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ([" DCM ", "nii.GZ", "dcm", ""], ["dcm", "nii.gz"]),
        (["png", "PNG", " jpeg "], ["png", "jpeg"]),
        ([], []),
    ],
)
def test_normalize_formats(raw, expected):
    assert normalize_formats(raw) == expected


@pytest.mark.parametrize(
    "query, expected_query, expected_formats",
    [
        (
            "segment lungs\nOriginalFormats: DCM NII.GZ\nwith contrast",
            "segment lungs with contrast",
            ["dcm", "nii.gz"],
        ),
        ("no legacy marker here", "no legacy marker here", []),
    ],
)
def test_strip_legacy_original_formats_line(query, expected_query, expected_formats):
    cleaned_query, formats = strip_legacy_original_formats_line(query)
    assert cleaned_query == expected_query
    assert formats == expected_formats


def test_append_format_tokens():
    out = append_format_tokens("segment lungs", ["dcm", "nii.gz"])
    assert "segment lungs" in out
    assert "format:DICOM" in out
    assert "format:NIfTI" in out
