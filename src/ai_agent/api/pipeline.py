# api/pipeline.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Optional

from ai_agent.retriever.reranker import CrossEncoderReranker
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.retriever.text_embedder import LocalBGEEmbedder
from ai_agent.retriever.vector_index import VectorIndex

from ai_agent.utils.tags import strip_tags
from ai_agent.utils.image_meta import detect_ext_token

log = logging.getLogger("pipeline")


class RAGImagingPipeline:
    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = Path(index_dir or os.getenv("RAG_INDEX_DIR", "artifacts/rag_index"))
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = LocalBGEEmbedder()
        self.reranker = CrossEncoderReranker()

        self.index = self._load_or_build_index()

    def _load_or_build_index(self) -> VectorIndex:
        try:
            return VectorIndex.load(self.index_dir, self.embedder)
        except Exception:
            return VectorIndex(self.embedder)

    def reload_index(self) -> bool:
        try:
            new_idx = VectorIndex.load(self.index_dir, self.embedder)
            self.index = new_idx
            return True
        except Exception:
            logging.getLogger("api").exception("reload_index failed")
            return False

    def _apply_reranker(self, query: str, hits: List[dict], top_k: int) -> List[dict]:
        if not hits:
            return []
        if self.reranker is None:
            return hits[:top_k]

        pool = hits[: min(len(hits), max(50, top_k * 3))]
        texts = [h["doc"].to_retrieval_text() for h in pool]

        try:
            ranked = self.reranker.rerank(query, texts, top_k=len(texts))
        except Exception:
            for h in pool:
                h["rerank_score"] = None
            return pool[:top_k]

        out: List[dict] = []
        for i, s in ranked[:top_k]:
            item = dict(pool[int(i)])
            item["rerank_score"] = float(s)
            out.append(item)

        if len(out) < top_k:
            used = {pool[int(i)]["id"] for i, _ in ranked[:top_k]}
            for h in pool:
                if h["id"] in used:
                    continue
                h = dict(h)
                h["rerank_score"] = h.get("rerank_score", None)
                out.append(h)
                if len(out) >= top_k:
                    break
        return out

    # ----------------------- Agent-facing lightweight APIs -------------------
    def retrieve_no_rerank(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
        top_k: int = 30,
        exclusions: Optional[List[str]] = None,
    ) -> List[dict]:
        """Return raw vector hits WITHOUT applying the CrossEncoder reranker.
        Each item: {id, doc, score}. Exclusions are case-insensitive on name.
        """
        def _norm(s: str) -> str:
            import re as _re
            return _re.sub(r"\s+", " ", (s or "").strip().lower())
        excluded_norm = {_norm(x) for x in (exclusions or []) if x}
        ext_tok = detect_ext_token(image_paths) if image_paths else ""
        clean_q = strip_tags(query)
        if ext_tok:
            clean_q = f"{clean_q} format:{ext_tok}" if clean_q else f"format:{ext_tok}"
        pool_k = max(50, top_k * 3)
        hits = self.index.search(clean_q, k=pool_k, reranker=None)
        if excluded_norm:
            hits = [h for h in hits if _norm(getattr(h["doc"], "name", "")) not in excluded_norm]
        # attach convenience fields expected downstream similar to recommend()
        for h in hits:
            h["__sim__"] = float(h.get("score", 0.0))
            h["__rerank__"] = 0.0
        return hits[:top_k]

    def rerank_only(self, query: str, hits: List[dict], top_k: int = 10) -> List[dict]:
        """Apply CrossEncoder reranker to a pre-fetched hit list.
        Returns new list of dicts (subset) with rerank_score set.
        """
        if not hits:
            return []
        # Recreate query with any existing format tokens already embedded in retrieval
        ranked = self._apply_reranker(strip_tags(query), hits, top_k=top_k)
        return ranked

    def get_doc(self, name: str) -> Optional[SoftwareDoc]:
        """Lookup a SoftwareDoc by name (case-sensitive match)."""
        try:
            return self.index.docs.get(name)
        except Exception:
            return None
