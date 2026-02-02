# api/pipeline.py
from __future__ import annotations

import os
import re 
import logging
from pathlib import Path
from typing import List, Optional

from ai_agent.retriever.reranker import CrossEncoderReranker
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.retriever.text_embedder import LocalBGEEmbedder
from ai_agent.retriever.vector_index import VectorIndex

from ai_agent.utils.tags import strip_tags
from ai_agent.utils.image_meta import detect_ext_token, summarize_image_metadata

log = logging.getLogger("pipeline")


class RAGImagingPipeline:
    def __init__(
        self,
        index_dir: Optional[str] = None,
        min_results: int = 5,
        max_retries: int = 2,
    ):
        """Initialize the RAG imaging pipeline."""
        self.index_dir = Path(index_dir or os.getenv("RAG_INDEX_DIR", "artifacts/rag_index"))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_results = min_results
        self.max_retries = max_retries

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
    def _build_image_hint_text(self, image_paths: Optional[List[str]]) -> str:
        """
        Turn image paths into extra text hints for retrieval.

        - Converts file extensions into format:xxx tokens (matching SoftwareDoc keywords)
        - Adds a short metadata summary (modality, body region, dims...)

        Result is a single string that we append to the text query before embedding.
        """
        if not image_paths:
            return ""

        hints: List[str] = []

        # 1) Format tokens (DICOM / NIfTI / TIFF / ...)
        ext_str = detect_ext_token(image_paths)
        if ext_str:
            for tok in ext_str.split():
                # match keywords like "format:tiff" that SoftwareDoc.to_retrieval_text()
                # puts into the index.
                hints.append(f"format:{tok.lower()}")

        # 2) Human-readable metadata (includes modality/body/dims)
        meta = summarize_image_metadata(image_paths)
        if meta:
            # collapse whitespace and keep it reasonably short
            compact = " ".join(meta.split())
            hints.append(compact[:300])

        return " ".join(hints)

    def retrieve_no_rerank(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
        top_k: int = 30,
        exclusions: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Return raw vector hits WITHOUT applying the CrossEncoder reranker.

        Each item: {id, doc, score}. Optional `image_paths` are used to derive
        additional text hints (format / modality / anatomy / dims) that are
        appended to the query before embedding.
        
        Relies on BGE-M3 semantic embeddings and approximate nearest-neighbor
        vector search.
        """

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip().lower())

        excluded_norm = {_norm(x) for x in (exclusions or []) if x}

        # 1) Strip any tags from the query
        clean_q = strip_tags(query)

        # 2) Add image-derived hints (format, modality, anatomy, dims, ...)
        image_hints = self._build_image_hint_text(image_paths)
        if image_hints:
            final_q = f"{clean_q} {image_hints}".strip()
        else:
            final_q = clean_q
        
        log.info(f"Retrieval query: {clean_q}" + (f" + metadata: {image_hints[:50]}..." if image_hints else ""))

        # 4) Vector search
        pool_k = max(50, top_k * 3)
        hits = self.index.search(final_q, k=pool_k, reranker=None)

        # 5) Apply name-based exclusions if any
        if excluded_norm:
            hits = [
                h
                for h in hits
                if _norm(getattr(h["doc"], "name", "")) not in excluded_norm
            ]

        # 6) Check if results are sufficient, retry with broader terms if not
        attempt = 0
        while len(hits) < self.min_results and attempt < self.max_retries:
            attempt += 1
            log.info(f"Insufficient results ({len(hits)} < {self.min_results}), attempting retry {attempt}/{self.max_retries}")
            
            # Generate alternative by simplifying query (remove specific terms, keep general ones)
            # Strategy: use first 2-3 words only to broaden the search
            words = clean_q.split()
            if len(words) > 3:
                alt_task = " ".join(words[:3])
                log.info(f"Trying broader query: {alt_task}")
                
                # Build alternative query with image hints
                if image_hints:
                    alt_q = f"{alt_task} {image_hints}".strip()
                else:
                    alt_q = alt_task
                
                # Search with alternative
                alt_hits = self.index.search(alt_q, k=pool_k, reranker=None)
                
                # Merge results (avoiding duplicates)
                existing_ids = {h["id"] for h in hits}
                for h in alt_hits:
                    if h["id"] not in existing_ids:
                        if not excluded_norm or _norm(getattr(h["doc"], "name", "")) not in excluded_norm:
                            hits.append(h)
                            existing_ids.add(h["id"])
                
                log.info(f"After retry {attempt}: {len(hits)} total results")
            else:
                log.warning(f"Query too short to generate alternative for retry {attempt}")
                break

        # 7) Attach convenience fields expected downstream
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
    
    def retrieve(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
        top_k: int = 10,
        exclusions: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Retrieve and automatically rerank results using BGE-M3 + CrossEncoder.
        
        This is the main retrieval method that combines:
        1. Semantic search via BGE-M3 embeddings (no query expansion)
        2. Precision reranking via CrossEncoder
        3. Image metadata hints (format, modality, dimensions)
        
        Returns top_k results after CrossEncoder reranking.
        """
        # Get more candidates than needed for reranking
        pool_k = max(30, top_k * 3)
        hits = self.retrieve_no_rerank(
            query=query,
            image_paths=image_paths,
            top_k=pool_k,
            exclusions=exclusions,
        )
        
        # Apply reranking to get final top_k
        if hits:
            return self.rerank_only(query, hits, top_k=top_k)
        return []

    def get_doc(self, name: str) -> Optional[SoftwareDoc]:
        """Lookup a SoftwareDoc by name (case-sensitive match)."""
        try:
            return self.index.docs.get(name)
        except Exception:
            return None
