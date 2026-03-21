from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from ai_agent.retriever.utils import _resolve_local_device

log = logging.getLogger("retriever.reranker")


class CrossEncoderReranker:
    """
    HTTP reranker client.

    Defaults to the EPFL OpenAI-compatible endpoint and BGE reranker model:
    - base_url: https://inference-rcp.epfl.ch/v1
    - model: BAAI/bge-reranker-v2-m3
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "https://inference-rcp.epfl.ch/v1",
        backend: str = "remote",
        api_key_env: str = "EPFL_API_KEY_EMBEDDER",
        timeout_s: float = 20.0,
        device: Optional[str] = None,
    ):
        self.backend = (backend or "remote").strip().lower()
        if self.backend not in {"remote", "local"}:
            raise ValueError(
                f"Unsupported reranker backend: {self.backend}. Use 'remote' or 'local'."
            )

        self.model_name = model_name

        if self.backend == "local":
            from sentence_transformers import CrossEncoder as STCrossEncoder

            resolved_device = _resolve_local_device(device)
            self._local_model = STCrossEncoder(
                self.model_name,
                device=resolved_device,
            )
            log.info(
                "Using local reranker backend with model=%s device=%s",
                self.model_name,
                resolved_device,
            )
            self.base_url = ""
            self.timeout_s = 0.0
            self.api_key_env = ""
            self.api_key = ""
            return

        self.base_url = (base_url or "").strip()
        if not self.base_url:
            raise ValueError("Reranker base_url must be provided for remote backend")
        self.base_url = self.base_url.rstrip("/")
        self.timeout_s = timeout_s

        self.api_key_env = api_key_env
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            log.warning(
                "%s is not set; reranking will be disabled and retrieval scores will be used.",
                api_key_env,
            )

    @staticmethod
    def _extract_rankings(payload: Dict[str, Any], doc_count: int) -> List[Tuple[int, float]]:
        """Parse API payload into sorted (index, score) tuples."""
        out: List[Tuple[int, float]] = []

        # Common shape: {"results": [{"index": 1, "relevance_score": 0.9}, ...]}
        results = payload.get("results")
        if isinstance(results, list):
            for row in results:
                if not isinstance(row, dict):
                    continue
                idx = row.get("index", row.get("document_index"))
                score = row.get("relevance_score", row.get("score"))
                if isinstance(idx, int) and 0 <= idx < doc_count and score is not None:
                    out.append((idx, float(score)))

        # Alternate shape: {"data": [{"index": 1, "score": 0.9}, ...]}
        if not out:
            data = payload.get("data")
            if isinstance(data, list):
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    idx = row.get("index", row.get("document_index"))
                    score = row.get("score", row.get("relevance_score"))
                    if isinstance(idx, int) and 0 <= idx < doc_count and score is not None:
                        out.append((idx, float(score)))

        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def rerank(
        self, query: str, texts: List[str], top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Returns list of (index_in_texts, score) sorted by score desc.
        """
        if not texts or top_k <= 0:
            return []

        if self.backend == "local":
            pairs = [(query, t) for t in texts]
            scores = self._local_model.predict(pairs)
            ranked = [(i, float(s)) for i, s in enumerate(scores)]
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked[: min(top_k, len(texts))]

        if not self.api_key:
            raise RuntimeError(
                f"Missing reranker API key in environment variable: {self.api_key_env}"
            )

        requested_top_k = min(top_k, len(texts))
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": requested_top_k,
            "return_documents": False,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()

        data = resp.json()
        ranked = self._extract_rankings(data, len(texts))
        if not ranked:
            raise RuntimeError(
                "Reranker endpoint returned no valid rankings. "
                f"Response keys: {sorted(data.keys()) if isinstance(data, dict) else type(data).__name__}"
            )
        return ranked[:requested_top_k]
