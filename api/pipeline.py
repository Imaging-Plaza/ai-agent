# api/pipeline.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from retriever.embedders import (
    LocalBGEEmbedder,
    VectorIndex,
    IndexItem,
    CrossEncoderReranker,
    SoftwareDoc,   # ensure SoftwareDoc has your fields; optional .runnables supported but not required
)
from generator.generator import VLMToolSelector
from generator.schema import CandidateDoc
from utils.image_meta import summarize_image_metadata, detect_ext_token

log = logging.getLogger("pipeline")


class RAGImagingPipeline:
    """
    Retrieval (text-only) + single-call VLM selection (image + text + candidates + metadata) + base runnable link.
    No server-side execution and no file deep-linking.
    """

    def __init__(self, docs: List[SoftwareDoc], workdir: str = "runs", hf_token: Optional[str] = None):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Build index
        self.embedder = LocalBGEEmbedder()
        self.index = VectorIndex(self.embedder)
        self.index.upsert([IndexItem(id=d.name, doc=d) for d in docs])

        self.reranker = CrossEncoderReranker()
        self.selector_vlm = VLMToolSelector()
        self.hf_token = hf_token  # unused in link-out mode; kept for future

        # Confidence gate (env)
        self.force_vlm = str(os.getenv("FORCE_VLM", "0")).lower() in ("1", "true", "yes", "on")
        self.conf_margin = float(os.getenv("RERANK_MARGIN", "0.15"))
        self.conf_top = float(os.getenv("RERANK_TOP", "0.90"))
        log.info(
            "Selector gate: FORCE_VLM=%s  RERANK_MARGIN=%.3f  RERANK_TOP=%.3f",
            self.force_vlm, self.conf_margin, self.conf_top
        )

    # -------- Retrieval (text-only, but add a light format token) --------
    def recommend(
        self, user_task: str, image_path: Optional[str], top_k: int = 5
    ) -> Tuple[List[dict], Dict[str, float]]:
        """
        Returns (hits, scores_summary).
        Retrieval uses only text, plus a deterministic 'format:EXT' token if an image path is provided.
        """
        ext_tok = detect_ext_token(image_path)  # e.g., "TIF", "NII.GZ"
        query = (user_task or "").strip()
        if ext_tok:
            query = f"{query} format:{ext_tok}"

        hits = self.index.search(query, k=20, reranker=self.reranker, rerank_top_k=top_k)

        # Log summary scores
        try:
            top = float(hits[0].get("rerank_score") or hits[0].get("score", 0.0)) if hits else 0.0
            second = float(hits[1].get("rerank_score") or hits[1].get("score", 0.0)) if len(hits) > 1 else 0.0
            margin = top - second
        except Exception:
            top = second = margin = 0.0

        log.info("Query: %s", query)
        log.info("Retrieval: top=%.3f  second=%.3f  margin=%.3f  (FORCE_VLM=%s)", top, second, margin, self.force_vlm)

        # Per-hit DEBUG
        for i, h in enumerate(hits, 1):
            sim = float(h.get("score", 0.0))
            rrs = float(h.get("rerank_score") or 0.0)
            name = getattr(h["doc"], "name", "?")
            log.debug("Hit %02d | name=%s | sim=%.3f | rerank=%.3f", i, name, sim, rrs)

        # Stash for downstream debugging/telemetry if needed
        for h in hits:
            h["__sim__"] = float(h.get("score", 0.0))
            h["__rerank__"] = float(h.get("rerank_score") or 0.0)

        return hits, {"top": top, "second": second, "margin": margin}

    # -------- Confidence gate --------
    def _is_confident(self, hits: List[dict]) -> bool:
        if self.force_vlm or not hits:
            return False
        top = float(hits[0].get("rerank_score") or hits[0].get("score", 0.0))
        second = float(hits[1].get("rerank_score") or hits[1].get("score", 0.0)) if len(hits) > 1 else 0.0
        return (top - second) > self.conf_margin or top >= self.conf_top

    # -------- Selection (0-call if confident, else 1-call VLM) --------
    def _select(self, user_task: str, hits: List[dict], image_path: Optional[str]) -> Tuple[dict, Dict[str, Any]]:
        candidates = [CandidateDoc(**h["doc"].model_dump()) for h in hits[:5]]

        # 0 calls: confident retrieval
        if self._is_confident(hits):
            chosen = hits[0]
            sel_json = {
                "choice": chosen["doc"].name,
                "alternates": [h["doc"].name for h in hits[1:4]],
                "why": "High-confidence retrieval; top candidate clearly dominates.",
            }
            return chosen, sel_json

        # 1 call: single-shot VLM (image + text + candidates + metadata)
        log.info("Using VLM selector (FORCE_VLM=%s)", self.force_vlm)
        try:
            meta_text = summarize_image_metadata(image_path) if image_path else None
            sel = self.selector_vlm.select(
                user_task=user_task,
                candidates=candidates,
                image_path=image_path,
                image_meta=meta_text,
            )
            chosen = next((h for h in hits if h["doc"].name == sel.choice), hits[0])
            return chosen, sel.model_dump()
        except Exception as e:
            log.warning("VLM selector failed (%s). Falling back to top-1.", e)
            chosen = hits[0]
            sel_json = {
                "choice": chosen["doc"].name,
                "alternates": [h["doc"].name for h in hits[1:4]],
                "why": "VLM unavailable; falling back to retrieval top-1.",
            }
            return chosen, sel_json

    # -------- Link-out (base URL only) --------
    def _best_runnable_link(self, doc: SoftwareDoc) -> Optional[str]:
        """
        Prefer catalog 'runnables' (lowest priority wins), else hf_space base URL, else None.
        """
        runnables = getattr(doc, "runnables", None) or []
        if runnables:
            r = sorted(runnables, key=lambda x: getattr(x, "priority", 100))[0]
            base = getattr(r, "url", None)
            if base:
                return base
        hf_space = getattr(doc, "hf_space", None)
        if hf_space:
            return f"https://huggingface.co/spaces/{hf_space}"
        return None

    # -------- Public API --------
    def recommend_and_link(self, image_path: Optional[str], user_task: str) -> Dict[str, Any]:
        hits, scores = self.recommend(user_task, image_path, top_k=5)
        if not hits:
            return {"error": "No candidates found."}

        chosen, selection = self._select(user_task, hits, image_path)
        doc: SoftwareDoc = chosen["doc"]
        link = self._best_runnable_link(doc)

        result: Dict[str, Any] = {
            "choice": doc.name,
            "why": selection.get("why", ""),
            "alternates": [h["doc"].name for h in hits if h["doc"].name != doc.name][:3],
            "scores": {k: round(v, 3) for k, v in scores.items()},
        }
        if link:
            result["demo_link"] = link
        return result
