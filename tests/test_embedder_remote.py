from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src" / "ai_agent"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ai_agent.retriever.text_embedder import LocalBGEEmbedder


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_embedder_uses_remote_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EPFL_API_KEY_EMBEDDER", "test-key")

    captured: dict = {}

    def _fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse(
            {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                    {"index": 1, "embedding": [0.0, 2.0, 0.0]},
                ]
            }
        )

    monkeypatch.setattr("ai_agent.retriever.text_embedder.requests.post", _fake_post)

    emb = LocalBGEEmbedder(
        backend="remote",
        model_name="Qwen/Qwen3-Embedding-8B",
        base_url="https://inference-rcp.epfl.ch/v1",
        api_key_env="EPFL_API_KEY_EMBEDDER",
    )
    vecs = emb.embed_corpus(["a", "b"])

    assert vecs.shape == (2, 3)
    assert emb.dim == 3
    assert captured["url"] == "https://inference-rcp.epfl.ch/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "Qwen/Qwen3-Embedding-8B"
    assert captured["json"]["input"][0].startswith("Represent the software for retrieval: ")

    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-6)


def test_embedder_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EPFL_API_KEY_EMBEDDER", raising=False)

    with pytest.raises(ValueError, match="API key not found"):
        LocalBGEEmbedder(api_key_env="EPFL_API_KEY_EMBEDDER")


def test_embedder_dim_infers_from_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EPFL_API_KEY_EMBEDDER", "test-key")

    def _fake_post(url, headers, json, timeout):
        _ = (url, headers, timeout)
        n = len(json["input"])
        data = [{"index": i, "embedding": [0.1, 0.2, 0.3, 0.4]} for i in range(n)]
        return _DummyResponse({"data": data})

    monkeypatch.setattr("ai_agent.retriever.text_embedder.requests.post", _fake_post)

    emb = LocalBGEEmbedder(api_key_env="EPFL_API_KEY_EMBEDDER")
    assert emb.dim == 4
