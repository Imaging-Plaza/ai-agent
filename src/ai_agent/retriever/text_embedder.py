from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

import numpy as np
import requests

from ai_agent.retriever.utils import _resolve_local_device


log = logging.getLogger("retriever.text_embedder")


class TextEmbedder:
    """Interface for text embeddings."""

    def embed_queries(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_corpus(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class LocalBGEEmbedder(TextEmbedder):
    """
    Remote embedder client (legacy class name retained for compatibility).

    Defaults to EPFL OpenAI-compatible embeddings endpoint and model:
    - base_url: https://inference-rcp.epfl.ch/v1
    - model: Qwen/Qwen3-Embedding-8B
    - api key env: EPFL_API_KEY_EMBEDDER
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: Optional[str] = None,
        backend: str = "remote",
        base_url: str = "https://inference-rcp.epfl.ch/v1",
        api_key_env: str = "EPFL_API_KEY_EMBEDDER",
        timeout_s: float = 20.0,
        dim: Optional[int] = None,
    ):
        self.backend = (backend or "remote").strip().lower()
        self.query_prefix = "Represent the query for retrieving relevant software: "
        self.doc_prefix = "Represent the software for retrieval: "
        self._dim: Optional[int] = dim

        if self.backend not in {"remote", "local"}:
            raise ValueError(
                f"Unsupported embedder backend: {self.backend}. Use 'remote' or 'local'."
            )

        if self.backend == "local":
            from sentence_transformers import SentenceTransformer

            self.model_name = model_name
            resolved_device = _resolve_local_device(device)
            self._local_model = SentenceTransformer(
                self.model_name,
                device=resolved_device,
            )
            self._dim = int(self._local_model.get_sentence_embedding_dimension())
            log.info(
                "Using local embedder backend with model=%s device=%s",
                self.model_name,
                resolved_device,
            )
            self.base_url = ""
            self.api_key_env = ""
            self.api_key = ""
            self.timeout_s = 0.0
            return

        # remote backend
        self.model_name = model_name
        self.base_url = (base_url or "").strip().rstrip("/")
        if not self.base_url:
            raise ValueError("Embedder base_url must be provided for remote backend")

        self.api_key_env = api_key_env
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            raise ValueError(
                f"API key not found in environment variable: {self.api_key_env}"
            )
        self.timeout_s = float(timeout_s)

    @property
    def dim(self) -> int:
        if self.backend == "local":
            return int(self._dim or 0)
        if self._dim is None:
            probe = self._encode(["dimension probe"])
            if probe.ndim != 2 or probe.shape[1] <= 0:
                raise RuntimeError("Failed to infer embedding dimension from endpoint response")
            self._dim = int(probe.shape[1])
        return int(self._dim)

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            if self._dim is None:
                return np.zeros((0, 0), dtype="float32")
            return np.zeros((0, self.dim), dtype="float32")

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": items,
            "encoding_format": "float",
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("Embedding endpoint returned invalid payload: missing 'data' list")

        pairs = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            idx = row.get("index")
            emb = row.get("embedding")
            if isinstance(idx, int) and isinstance(emb, list):
                pairs.append((idx, emb))

        if len(pairs) != len(items):
            raise RuntimeError(
                f"Embedding endpoint returned {len(pairs)} vectors for {len(items)} inputs"
            )

        pairs.sort(key=lambda x: x[0])
        vecs = np.asarray([emb for _, emb in pairs], dtype="float32")
        if vecs.ndim != 2:
            raise RuntimeError("Embedding endpoint returned malformed vectors")

        # Normalize embeddings for cosine/IP consistency with FAISS IndexFlatIP.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        vecs = vecs / norms

        if self._dim is None:
            self._dim = int(vecs.shape[1])
        elif int(vecs.shape[1]) != int(self._dim):
            raise RuntimeError(
                f"Embedding dimension changed from {self._dim} to {vecs.shape[1]}"
            )

        return vecs.astype("float32")

    def embed_queries(self, texts: Iterable[str]) -> np.ndarray:
        if self.backend == "local":
            arr = self._local_model.encode(
                [self.query_prefix + t for t in texts],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(arr, dtype="float32")
        return self._encode([self.query_prefix + t for t in texts])

    def embed_corpus(self, texts: Iterable[str]) -> np.ndarray:
        if self.backend == "local":
            arr = self._local_model.encode(
                [self.doc_prefix + t for t in texts],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(arr, dtype="float32")
        return self._encode([self.doc_prefix + t for t in texts])
