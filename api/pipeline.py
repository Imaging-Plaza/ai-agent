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
    SoftwareDoc,   # ensure SoftwareDoc maps your catalog; optional .runnables supported
)
from generator.generator import VLMToolSelector
from generator.schema import CandidateDoc
from utils.image_meta import summarize_image_metadata, detect_ext_token

log = logging.getLogger("pipeline")


class RAGImagingPipeline:
    """
    Retrieval (text-only) + single-call VLM selection (image + text + candidates + metadata) + base runnable link.
    Persists FAISS to disk and keeps it in sync with catalog changes.
    """

    def __init__(
        self,
        docs: List[SoftwareDoc],
        hf_token: Optional[str] = None,
        index_dir: Optional[str] = None,
    ):
        # --------- Config ---------
        self.index_dir = Path(index_dir or os.getenv("RAG_INDEX_DIR", "artifacts/rag_index"))
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # --------- Models ---------
        self.embedder = LocalBGEEmbedder()
        self.reranker = CrossEncoderReranker()
        self.selector_vlm = VLMToolSelector()
        self.hf_token = hf_token  # unused in link-out mode; kept for future

        # --------- Confidence gate (env) ---------
        self.force_vlm = str(os.getenv("FORCE_VLM", "0")).lower() in ("1", "true", "yes", "on")
        self.conf_margin = float(os.getenv("RERANK_MARGIN", "0.15"))
        self.conf_top = float(os.getenv("RERANK_TOP", "0.90"))
        log.info(
            "Selector gate: FORCE_VLM=%s  RERANK_MARGIN=%.3f  RERANK_TOP=%.3f",
            self.force_vlm, self.conf_margin, self.conf_top
        )

        # --------- Index: load or build, then sync with provided docs ---------
        self.index = self._load_or_build_index()
        if docs:
            # keep ID stable; if name can change, swap to a durable field
            stats = self.index.sync_with_catalog([IndexItem(id=d.name, doc=d) for d in docs])
            if any(stats.values()):
                self.index.save(self.index_dir)
                log.info("Index built/synced at %s: %s", self.index_dir, stats)
            else:
                log.info("Catalog identical to saved index — no changes.")

    # ---------- Index lifecycle ----------

    def _load_or_build_index(self) -> VectorIndex:
        try:
            idx = VectorIndex.load(self.index_dir, self.embedder)
            log.info("Loaded FAISS index from %s (ntotal=%d)", self.index_dir, idx._index.ntotal)
            return idx
        except (FileNotFoundError, RuntimeError) as e:
            log.warning("No saved index yet (%s) — creating a new empty index.", e)
        except ValueError as e:
            # e.g., embedder dim/prefix mismatch
            log.warning("Saved index incompatible (%s) — creating a new empty index.", e)

        return VectorIndex(self.embedder)

    def refresh_catalog(self, docs: List[SoftwareDoc]) -> Dict[str, int]:
        """
        Call when your catalog changes (adds/updates/deletes).
        Diffs against current index; re-embeds only what's changed; saves if needed.
        """
        items = [IndexItem(id=d.name, doc=d) for d in docs]
        stats = self.index.sync_with_catalog(items)
        if any(stats.values()):
            self.index.save(self.index_dir)
            log.info("Index updated & saved at %s: %s", self.index_dir, stats)
        else:
            log.info("Catalog unchanged — index not modified.")
        return stats

    # -------- Internal: apply CrossEncoder reranker robustly --------
    def _apply_reranker(self, query: str, hits: List[dict], top_k: int) -> List[dict]:
        if not hits:
            return []
        if self.reranker is None:
            return hits[:top_k]

        pool = hits[: min(len(hits), max(50, top_k * 3))]
        texts = [h["doc"].to_retrieval_text() for h in pool]

        try:
            ranked = self.reranker.rerank(query, texts, top_k=len(texts))
        except Exception as e:
            log.warning("Reranker call failed (%s). Falling back to vector ranking.", e)
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

    # -------- Retrieval (text-only, optional format token) --------
    def recommend(
        self, user_task: str, image_path: Optional[str], top_k: int = 5
    ) -> Tuple[List[dict], Dict[str, float]]:
        ext_tok = detect_ext_token(image_path)  # e.g., "TIF", "NII.GZ"
        query = (user_task or "").strip()
        if ext_tok:
            query = f"{query} format:{ext_tok}"

        raw_hits = self.index.search(query, k=50, reranker=None)
        hits = self._apply_reranker(query, raw_hits, top_k=top_k)

        try:
            top = float(hits[0].get("rerank_score") or hits[0].get("score", 0.0)) if hits else 0.0
            second = float(hits[1].get("rerank_score") or hits[1].get("score", 0.0)) if len(hits) > 1 else 0.0
            margin = top - second
        except Exception:
            top = second = margin = 0.0

        log.info("Query: %s", query)
        log.info("Retrieval: top=%.3f  second=%.3f  margin=%.3f  (FORCE_VLM=%s)", top, second, margin, self.force_vlm)

        for i, h in enumerate(hits, 1):
            sim = float(h.get("score", 0.0))
            rrs = float(h.get("rerank_score") or 0.0)
            name = getattr(h["doc"], "name", "?")
            log.debug("Hit %02d | name=%s | sim=%.3f | rerank=%.3f", i, name, sim, rrs)

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
    def _select(
        self, user_task: str, hits: List[dict], image_path: Optional[str]
    ) -> Tuple[Optional[dict], Dict[str, Any]]:  # <-- return Optional[dict]
        # Build candidates safely (limit to first 5 hits)
        candidates = []
        for h in hits[:5]:
            try:
                candidates.append(CandidateDoc(**h["doc"].model_dump()))
            except Exception:
                continue

        # Quick exit if retrieval already looks confident
        if self._is_confident(hits):
            chosen = hits[0]
            sel_json = {
                "choice": chosen["doc"].name,
                "alternates": [h["doc"].name for h in hits[1:4]],
                "why": "High-confidence retrieval; top candidate clearly dominates.",
            }
            return chosen, sel_json

        log.info("Using VLM selector (FORCE_VLM=%s)", getattr(self, "force_vlm", None))

        meta_text = None
        try:
            meta_text = summarize_image_metadata(image_path) if image_path else None
        except Exception:
            log.exception("Image metadata summarization failed; continuing without metadata.")

        try:
            sel = self.selector_vlm.select(
                user_task=user_task,
                candidates=candidates,
                image_path=image_path,
                image_meta=meta_text,
            )
            sel_json = sel.model_dump()

            # If VLM explicitly said "none", propagate that
            if (sel.choice or "").strip().lower() == "none":
                return None, sel_json

            # Otherwise map to a hit; if not found, fallback to top-1
            chosen = next((h for h in hits if h["doc"].name == sel.choice), hits[0])
            return chosen, sel_json

        except Exception as e:
            log.exception("VLM selector failed. Falling back to top-1.")
            chosen = hits[0]
            alternates = [h["doc"].name for h in hits[1:4]]
            log_path_note = ""
            try:
                lf = getattr(self.selector_vlm, "last_logfile", None)
                if lf:
                    log_path_note = f" (prompt log: {lf})"
            except Exception:
                pass

            sel_json = {
                "choice": chosen["doc"].name,
                "alternates": alternates,
                "why": f"VLM unavailable or errored ('{e.__class__.__name__}: {e}'); "
                    f"falling back to retrieval top-1{log_path_note}.",
            }
            return chosen, sel_json


    # -------- Link-out (base URL only) --------
    def _best_runnable_link(self, doc: SoftwareDoc) -> Optional[str]:
        """
        1) Return best URL from `runnable_example` (alias of dataset `runnableExample`).
        2) Else return best URL from `has_executable_notebook` (alias `hasExecutableNotebook`).
        3) Else None.
        If items include a `priority` field, lower wins; otherwise preserve original order.
        """

        def priority(item) -> float:
            if isinstance(item, dict) and "priority" in item:
                try:
                    return float(item["priority"])
                except Exception:
                    pass
            return 1e9  # stable sort keeps original order among equal priorities

        def extract_url(item) -> Optional[str]:
            if isinstance(item, str):
                u = item.strip()
                return u or None
            if isinstance(item, dict):
                for k in ("url", "href", "link", "contentUrl"):
                    u = item.get(k)
                    if isinstance(u, str) and u.strip():
                        return u.strip()
            return None

        # Preference order: runnable example first, then executable notebook
        for items in (doc.runnable_example or [], doc.has_executable_notebook or []):
            try:
                items_sorted = sorted(items, key=priority)
            except Exception:
                items_sorted = items
            for it in items_sorted:
                url = extract_url(it)
                if url:
                    return url

        return None

    # -------- Public API --------
    def recommend_and_link(self, image_path: Optional[str], user_task: str) -> Dict[str, Any]:
        hits, scores = self.recommend(user_task, image_path, top_k=5)
        if not hits:
            return {"error": "No candidates found."}

        chosen, selection = self._select(user_task, hits, image_path)

        if chosen is None or (selection.get("choice", "").strip().lower() == "none"):
            return {
                "choice": "none",
                "why": selection.get("why", "No suitable tool among candidates."),
                "alternates": [],
                "scores": {k: round(v, 3) for k, v in scores.items()},
            }

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

