# retriever/embedders.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional

import faiss                   # pip install faiss-cpu
import numpy as np             # pip install numpy
from sentence_transformers import SentenceTransformer, CrossEncoder  # pip install sentence-transformers
from pydantic import BaseModel, Field

# ----------------------------
# Data model (keep in sync with your catalog schema)
# ----------------------------

class SoftwareDoc(BaseModel):
    name: str
    repo_url: Optional[str] = None
    tasks: List[str] = Field(default_factory=list)
    modality: List[str] = Field(default_factory=list)
    dims: List[str] = Field(default_factory=list)
    #anatomy: List[str] = Field(default_factory=list)
    input_formats: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    #install_cmd: Optional[str] = None
    weights_available: Optional[bool] = None
    license: Optional[str] = None
    os: List[str] = Field(default_factory=list)
    gpu_required: Optional[bool] = None
    #last_commit_date: Optional[str] = None
    #sample_snippet: Optional[str] = None
    # raw text fallback (if you store long descriptions)
    description: Optional[str] = None
    hf_space: Optional[str] = None        # e.g., "owner/space-name"
    hf_api_name: Optional[str] = None     # e.g., "/predict" (gradio api_name)
    space_input: Optional[str] = "file"     # "file" | "image" | "url"
    space_params: Dict[str, Any] = Field(default_factory=dict)
    space_timeout: Optional[int] = 600
    hf_calls: List[Dict[str, Any]] = Field(default_factory=list)

    def to_retrieval_text(self) -> str:
        """Compact serialization for embedding/retrieval."""
        fields = [
            f"name: {self.name}",
            f"tasks: {', '.join(self.tasks)}",
            f"modality: {', '.join(self.modality)}",
            f"dims: {', '.join(self.dims)}",
            #f"anatomy: {', '.join(self.anatomy)}",
            f"inputs: {', '.join(self.input_formats)}",
            f"outputs: {', '.join(self.output_types)}",
            f"language: {self.language or ''}",
            f"license: {self.license or ''}",
            f"gpu_required: {self.gpu_required}",
            f"os: {', '.join(self.os)}",
            f"weights_available: {self.weights_available}",
            #f"last_commit: {self.last_commit_date or ''}",
            #f"snippet: {self.sample_snippet or ''}",
            f"desc: {self.description or ''}",
        ]
        return " | ".join(fields)


# ----------------------------
# Embedders
# ----------------------------

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

        # BGE works best with small instruction prefixes; keep them tweakable.
        self.query_prefix = "Represent the query for retrieving relevant software: "
        self.doc_prefix = "Represent the software for retrieval: "

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            normalize_embeddings=True,  # for cosine/IP search
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def embed_queries(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def embed_corpus(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])


# ----------------------------
# Reranker (Cross-Encoder)
# ----------------------------

class CrossEncoderReranker:
    """
    Strong re-ranker. Default: BAAI/bge-reranker-v2-m3 (multilingual).
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        """
        Returns list of (index_in_texts, score) sorted by score desc.
        """
        pairs = [[query, t] for t in texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[int(i)])) for i in order]


# ----------------------------
# FAISS index wrapper
# ----------------------------

@dataclass
class IndexItem:
    id: str
    doc: SoftwareDoc

class VectorIndex:
    """
    Cosine-similarity FAISS index (via inner-product on normalized vectors).
    Stores metadata in-memory (ids->docs). For persistence, save/reload the faiss
    index plus a sidecar metadata JSON.
    """
    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder
        self.ids: List[str] = []
        self.docs: List[SoftwareDoc] = []
        self._index = faiss.IndexFlatIP(embedder.dim)  # inner product on normalized vectors
        self._matrix: Optional[np.ndarray] = None      # keep a copy for incremental adds

    def _ensure_matrix(self):
        if self._matrix is None:
            self._matrix = np.zeros((0, self.embedder.dim), dtype="float32")

    def upsert(self, items: List[IndexItem]) -> None:
        self._ensure_matrix()
        texts = [it.doc.to_retrieval_text() for it in items]
        embs = self.embedder.embed_corpus(texts)
        # Append
        self._matrix = np.vstack([self._matrix, embs])
        self._index.add(embs)
        self.ids.extend([it.id for it in items])
        self.docs.extend([it.doc for it in items])

    def search(
        self,
        query_text: str,
        k: int = 20,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: {id, doc, score, rerank_score (opt)}
        """
        q = self.embedder.embed_queries([query_text])
        D, I = self._index.search(q, min(k, len(self.ids)))
        scores = D[0].tolist()
        idxs = I[0].tolist()

        hits = [
            {"id": self.ids[i], "doc": self.docs[i], "score": float(scores[j])}
            for j, i in enumerate(idxs) if i != -1
        ]
        if reranker and hits:
            texts = [h["doc"].to_retrieval_text() for h in hits]
            reranked = reranker.rerank(query_text, texts, top_k=min(rerank_top_k, len(hits)))
            # Map rerank order back
            ranked = [hits[i] | {"rerank_score": s} for i, s in reranked]
            return ranked
        return hits
