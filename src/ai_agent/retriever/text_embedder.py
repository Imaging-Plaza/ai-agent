from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


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
    BGE-M3 embedder (OSS). Good default for retrieval.
    Model card: BAAI/bge-m3
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        self.model = SentenceTransformer(model_name, device=device)
        self._dim = self.model.get_sentence_embedding_dimension()
        self.query_prefix = "Represent the query for retrieving relevant software: "
        self.doc_prefix = "Represent the software for retrieval: "

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def embed_queries(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def embed_corpus(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])
