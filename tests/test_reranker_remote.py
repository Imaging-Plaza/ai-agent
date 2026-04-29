from __future__ import annotations

from pathlib import Path
import sys

import pytest

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src" / "ai_agent"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ai_agent.retriever.reranker import CrossEncoderReranker


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_extract_rankings_results_shape() -> None:
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.2},
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
        ]
    }
    out = CrossEncoderReranker._extract_rankings(payload, doc_count=3)
    assert out == [(0, 0.9), (1, 0.5), (2, 0.2)]


def test_rerank_uses_remote_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EPFL_API_KEY_EMBEDDER", "test-key")

    captured: dict = {}

    def _fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 0, "relevance_score": 0.1},
                ]
            }
        )

    monkeypatch.setattr("ai_agent.retriever.reranker.requests.post", _fake_post)

    rr = CrossEncoderReranker(
        backend="remote",
        model_name="BAAI/bge-reranker-v2-m3",
        base_url="https://inference-rcp.epfl.ch/v1",
        api_key_env="EPFL_API_KEY_EMBEDDER",
    )
    ranked = rr.rerank("segment lungs", ["doc a", "doc b"], top_k=2)

    assert ranked == [(1, 0.8), (0, 0.1)]
    assert captured["url"] == "https://inference-rcp.epfl.ch/v1/rerank"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert captured["json"]["top_n"] == 2


def test_rerank_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EPFL_API_KEY_EMBEDDER", raising=False)
    rr = CrossEncoderReranker(api_key_env="EPFL_API_KEY_EMBEDDER")
    with pytest.raises(RuntimeError, match="Missing reranker API key"):
        rr.rerank("q", ["a"], top_k=1)
