from __future__ import annotations

import json
from pathlib import Path
import sys

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src" / "ai_agent"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ai_agent.api.pipeline import RAGImagingPipeline


class _DummyPipeline:
    _read_catalog_docs = RAGImagingPipeline._read_catalog_docs


def test_read_catalog_docs_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "catalog.jsonl"
    rows = [
        {"name": "tool-a", "description": "A"},
        {"name": "tool-b", "description": "B"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    docs = _DummyPipeline()._read_catalog_docs(p)
    assert [d.name for d in docs] == ["tool-a", "tool-b"]


def test_read_catalog_docs_json_array(tmp_path: Path) -> None:
    p = tmp_path / "catalog.json"
    rows = [
        {"name": "tool-a", "description": "A"},
        {"name": "tool-b", "description": "B"},
    ]
    p.write_text(json.dumps(rows), encoding="utf-8")

    docs = _DummyPipeline()._read_catalog_docs(p)
    assert [d.name for d in docs] == ["tool-a", "tool-b"]
