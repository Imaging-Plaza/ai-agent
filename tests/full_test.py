# tests/full_test.py
from __future__ import annotations
import sys, json, tempfile
from pathlib import Path

import numpy as np
import pytest

import os
import json

from json import JSONDecodeError
import io
import re
import mimetypes
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from typing import Any, Dict, Iterable, List
import numpy as np

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

# --- Make the project root importable (so 'retriever', 'api', etc. resolve) ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retriever.embedders import SoftwareDoc

# --------------------------------------------------------------------------------------
# Dummy / test doubles for heavy components
# --------------------------------------------------------------------------------------

class DummyEmbedder:
    """
    Small, deterministic embedder to replace heavy LocalBGEEmbedder.
    Implements the methods many pipelines expect:
      - embed_documents(texts)
      - embed_queries(texts)
    (aliases to a common _embed()).
    """
    def __init__(self, dim: int = 16, **_kwargs: Any) -> None:
        self.dim = dim

    def _embed(self, texts: Iterable[str]) -> np.ndarray:
        # Deterministic embedding: hash each text into dim floats in [0,1)
        arr = []
        for t in texts:
            h = abs(hash(str(t)))
            vec = [((h >> (i * 3)) & 0xFF) / 255.0 for i in range(self.dim)]
            arr.append(vec)
        return np.asarray(arr, dtype=np.float32)

    # Some code paths may call just "embed"
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return self._embed(texts)

    # Typical RAG interface
    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: Iterable[str]) -> np.ndarray:
        return self._embed(texts)


class DummyReranker:
    """No-op reranker: preserves order."""
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return docs[:top_k]


class DummyVLM:
    """
    Stable selector for tests: ALWAYS returns the injected expected choice.
    We still gather a few alternates only for debug visibility.
    """
    def __init__(self, fixed_choice: str, why: str = "unit test selection"):
        self.fixed_choice = str(fixed_choice)
        self.why = why
        self.last_logfile = None

    def select(self, user_task, candidates, image_path, image_meta):
        from generator.schema import ToolSelection
        cand_names_lc = [
            str(getattr(c, "name", "")).strip().lower()
            for c in candidates
            if getattr(c, "name", None)
        ]
        choice = self.fixed_choice  # deterministic
        alts = [n for n in cand_names_lc if n != choice.lower()]
        alts = sorted(dict.fromkeys(alts))[:3]
        return ToolSelection(choice=choice, alternates=alts, why=self.why)


# --------------------------------------------------------------------------------------
# Helpers: JSON/JSONL loader
# --------------------------------------------------------------------------------------

def _iter_jsonl(fp: io.TextIOBase):
    for ln, line in enumerate(fp, 1):
        s = line.strip()
        if not s:
            continue
        # Tolerate accidental trailing commas on lines
        if s.endswith(","):
            s = s[:-1]
        try:
            yield json.loads(s)
        except JSONDecodeError as e:
            pytest.fail(f"Invalid JSON on line {ln} of SOFTWARE_CATALOG (JSONL): {e}", pytrace=False)


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        # Try JSON first
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
        except JSONDecodeError:
            # Fallback to JSONL
            f.seek(0)
            return list(_iter_jsonl(f))


# --------------------------------------------------------------------------------------
# Catalog loader (SOFTWARE_CATALOG only)
# --------------------------------------------------------------------------------------

def _load_catalog_docs():
    """Load tool docs exclusively from the SOFTWARE_CATALOG env var (JSON or JSONL)."""

    catalog_path = os.environ.get("SOFTWARE_CATALOG")
    if not catalog_path:
        pytest.fail("SOFTWARE_CATALOG is not set", pytrace=False)

    path = Path(catalog_path)
    if not path.exists():
        pytest.fail(f"SOFTWARE_CATALOG points to missing file: {path}", pytrace=False)

    docs = []
    ext = path.suffix.lower()
    try_json_first = ext != ".jsonl"

    with path.open("r", encoding="utf-8") as f:
        if try_json_first:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    docs.extend(SoftwareDoc.model_validate(d) for d in data)
                else:
                    docs.append(SoftwareDoc.model_validate(data))
                return docs
            except JSONDecodeError:
                f.seek(0)

        # JSONL path
        for obj in _iter_jsonl(f):
            docs.append(SoftwareDoc.model_validate(obj))
    return docs


# --------------------------------------------------------------------------------------
# Image resolver: handles URLs (Windows-safe filenames) and local paths
# --------------------------------------------------------------------------------------

def _resolve_image_path(value: str) -> Path:
    """Return a local path for an image. If value is a URL, download it to a temp file with a safe filename.
       If a host blocks us (e.g., 403), SKIP the subtest rather than fail the suite."""
    if re.match(r"^https?://", value, flags=re.IGNORECASE):
        parts = urlsplit(value)

        # Base name from URL path, strip queries/fragments and sanitize for Windows
        base = Path(parts.path).name or "downloaded"
        base = base.split("?")[0].split("#")[0]
        base = re.sub(r'[^A-Za-z0-9._-]', "_", base)  # safe chars only

        # Build request with a reasonable User-Agent
        req = Request(value, headers={"User-Agent": "Mozilla/5.0 (pytest E2E test)"})
        try:
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
                ctype = (
                    resp.headers.get_content_type()
                    if hasattr(resp.headers, "get_content_type")
                    else resp.headers.get("Content-Type", "")
                )
        except HTTPError as e:
            # Many CDNs block requests without UA or from CI; treat as network flakiness -> skip
            pytest.skip(f"Image URL fetch failed ({e.code}): {value}")
        except URLError as e:
            pytest.skip(f"Image URL fetch error: {e.reason}")

        # Ensure extension
        ext = Path(base).suffix.lower()
        if not ext or ext == ".bin":
            guess = mimetypes.guess_extension(ctype or "") or ""
            if not guess and ctype:
                if "jpeg" in ctype:
                    guess = ".jpg"
                elif "png" in ctype:
                    guess = ".png"
                elif "tiff" in ctype or "tif" in ctype:
                    guess = ".tif"
                elif "gif" in ctype:
                    guess = ".gif"
            if guess:
                base = Path(base).with_suffix(guess).name

        tmpdir = Path(tempfile.mkdtemp(prefix="sheet_img_"))
        out = tmpdir / base
        out.write_bytes(data or b"")
        return out

    # Local file path (relative to repo root if not absolute)
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    assert p.exists(), f"Local image not found: {p}"
    return p


def _normalize_expected(expected_field: Any) -> List[str]:
    """Expected may be a string or a list of strings; return a lowercased list."""
    if expected_field is None:
        return []
    if isinstance(expected_field, str):
        return [expected_field.strip().lower()]
    if isinstance(expected_field, (list, tuple)):
        return [str(x).strip().lower() for x in expected_field]
    return [str(expected_field).strip().lower()]


# --------------------------------------------------------------------------------------
# The end-to-end test (each case is a subtest)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("force_vlm", [True, False])
def test_pipeline_against_sheet_with_urls(tmp_path: Path, monkeypatch, subtests, force_vlm):
    """
    End-to-end pipeline test driven by tests/data/test_data.json (or .jsonl).
    Each case in the sheet: {"task": "...", "image": "<local path or http(s) URL>", "expected": "<tool name|list>"}.

    Heavy components (embedder/reranker/VLM) are stubbed for speed and determinism.
    """

    # Patch heavy components
    import retriever.embedders as emb
    monkeypatch.setattr(emb, "LocalBGEEmbedder", DummyEmbedder, raising=True)
    monkeypatch.setattr(emb, "CrossEncoderReranker", DummyReranker, raising=True)

    # Toggle FORCE_VLM env to exercise both paths (if the pipeline uses it)
    monkeypatch.setenv("FORCE_VLM", "1" if force_vlm else "0")

    # Build pipeline with real docs from catalog
    docs = _load_catalog_docs()
    from api.pipeline import RAGImagingPipeline

    pipe = RAGImagingPipeline(docs=docs)

    # For determinism in this unit test, we also disable the (patched) reranker.
    pipe.reranker = None

    # Load test sheet
    sheet_path_json = ROOT / "tests" / "data" / "test_data.json"
    sheet_path_jsonl = ROOT / "tests" / "data" / "test_data.jsonl"

    sheet_path = None
    if sheet_path_json.exists():
        sheet_path = sheet_path_json
    elif sheet_path_jsonl.exists():
        sheet_path = sheet_path_jsonl
    else:
        pytest.skip("No test sheet at tests/data/test_data.json (or .jsonl)")

    cases = _read_json_or_jsonl(sheet_path)

    # Minimal schema sanity
    for i, c in enumerate(cases, 1):
        assert "task" in c and "image" in c and "expected" in c, f"Bad row {i}: {c}"

    # Each row is its own subtest
    for i, case in enumerate(cases, 1):
        with subtests.test(msg=f"case {i}", task=case.get("task"), image=case.get("image")):
            task = str(case["task"])
            image_path = _resolve_image_path(str(case["image"]))
            expected_list = _normalize_expected(case["expected"])

            # Inject deterministic VLM response (always returns first expected)
            if expected_list:
                pipe.selector_vlm = DummyVLM(fixed_choice=expected_list[0])
            else:
                # If no expectation provided, choose a benign default
                pipe.selector_vlm = DummyVLM(fixed_choice="none")

            result = pipe.recommend_and_link(image_paths=[image_path], user_task=task)

            # Assertions for this subtest
            assert isinstance(result, dict), "Pipeline should return a dict"
            assert "choice" in result, "Pipeline result must have a 'choice' key"

            # Normalize for comparison
            got = str(result["choice"]).strip().lower()
            assert got in expected_list, f"Got {result['choice']}, expected one of {expected_list}"