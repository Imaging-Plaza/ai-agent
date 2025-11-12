# api/pipeline.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re
from utils.tags import strip_tags, parse_exclusions, has_no_rerank

from retriever.reranker import CrossEncoderReranker
from retriever.software_doc import SoftwareDoc
from retriever.text_embedder import LocalBGEEmbedder
from retriever.vector_index import VectorIndex
from generator.generator import VLMToolSelector
from generator.schema import CandidateDoc, NoToolReason
from utils.file_validator import FileValidator

from utils.image_meta import detect_ext_token
from utils.previews import _build_preview_for_vlm, _cleanup_old_previews
from utils.utils import _best_runnable_link

log = logging.getLogger("pipeline")


class RAGImagingPipeline:
    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = Path(index_dir or os.getenv("RAG_INDEX_DIR", "artifacts/rag_index"))
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = LocalBGEEmbedder()
        self.reranker = CrossEncoderReranker()
        self.selector_vlm = VLMToolSelector()

        try:
            _cleanup_old_previews(hours=24)
        except Exception:
            logging.getLogger("api").exception("Preview cleanup at init failed; continuing")

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

    def recommend(self, user_task: str, image_paths: Optional[List[str]], top_k: int = 5,
                persisted_exclusions: Optional[List[str]] = None
        ) -> Tuple[List[dict], Dict[str, float]]:

        """
        Retrieve candidate tools for the given request. Control tags:
        [NO_RERANK]      -> skip CrossEncoder reranker
        [EXCLUDE:a|b]    -> exclude these tool names *before* reranking/top-k
        [EXCLUDED:a|b]   -> alias of EXCLUDE
        """
        def _norm(s: str) -> str:
            # normalize: lowercase, trim, collapse whitespace
            return re.sub(r"\s+", " ", (s or "").strip().lower())

        # --- Control tags ---------------------------------------------------------
        skip_rerank = has_no_rerank(user_task)
        excluded_raw = set(parse_exclusions(user_task))
        if persisted_exclusions:
            excluded_raw |= set(persisted_exclusions)
        excluded_norm = {_norm(x) for x in excluded_raw}

        # Work with a clean task (no control tags) for retrieval
        clean_task = strip_tags(user_task)

        # --- Build retrieval query ------------------------------------------------
        ext_tok = detect_ext_token(image_paths)  # e.g., "DICOM NIfTI TIFF"
        query = (clean_task or "").strip()
        if ext_tok:
            query = f"{query} format:{ext_tok}"

        # --- Vector search (generous pool) ---------------------------------------
        pool_k = max(50, top_k * 3)
        raw_hits = self.index.search(query, k=pool_k, reranker=None)

        # Exclude BEFORE reranking (normalized name compare)
        if excluded_norm:
            _before = len(raw_hits)
            raw_hits = [
                h for h in raw_hits
                if _norm(getattr(h["doc"], "name", "")) not in excluded_norm
            ]
            log.info("Excluded %d/%d by name (norm)", _before - len(raw_hits), _before)

        # --- Optional reranker ----------------------------------------------------
        if skip_rerank:
            hits = raw_hits[:top_k]
            for h in hits:
                h.setdefault("score", h.get("score", 0.0))
                h["rerank_score"] = None
        else:
            hits = self._apply_reranker(query, raw_hits, top_k=top_k)

        # --- Score summary for telemetry/UX --------------------------------------
        try:
            top = float(hits[0].get("rerank_score") or hits[0].get("score", 0.0)) if hits else 0.0
            second = float(hits[1].get("rerank_score") or hits[1].get("score", 0.0)) if len(hits) > 1 else 0.0
            margin = top - second
        except Exception:
            top = second = margin = 0.0

        # Summary log (preserved + show excluded count)
        log.info(
            "Retrieval: top=%.3f  second=%.3f  margin=%.3f  skip_rerank=%s  excluded=%d",
            top, second, margin, "YES" if skip_rerank else "NO", len(excluded_norm)
        )

        # Per-hit debug log (preserved)
        for i, h in enumerate(hits, 1):
            name = getattr(h["doc"], "name", "?")
            sim = float(h.get("score", 0.0))
            rrs = float(h.get("rerank_score") or 0.0)
            log.debug("Hit %02d | name=%s | sim=%.3f | rerank=%.3f", i, name, sim, rrs)

        # Attach normalized convenience fields used later
        for h in hits:
            h["__sim__"] = float(h.get("score", 0.0))
            h["__rerank__"] = float(h.get("rerank_score") or 0.0)

        return hits, {"top": top, "second": second, "margin": margin}

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


    def _select(self, hits, image_meta_text, user_task, preview_path):
        num_choices = int(os.getenv("NUM_CHOICES", "3"))

        candidates = []
        for h in hits:
            d = h["doc"]
            try:
                cd = CandidateDoc.model_validate(d.model_dump(mode="python"))
            except Exception:
                cd = CandidateDoc(
                    name=getattr(d, "name", None),
                    description=getattr(d, "description", None),
                    url=getattr(d, "url", None),
                    tasks=getattr(d, "tasks", []),
                    modality=getattr(d, "modality", []),
                    dims=getattr(d, "dims", []),
                    programming_language=getattr(d, "programming_language", None),
                    gpu_required=getattr(d, "gpu", None),
                    runnable_examples=getattr(d, "runnable_example", []),
                    executable_notebooks=getattr(d, "has_executable_notebook", []),
                )
            candidates.append(cd)

        sel = self.selector_vlm.select(
            user_task=user_task,
            candidates=candidates,
            image_path=preview_path,
            image_meta=image_meta_text or "",
        )

        sel_json = sel.model_dump(mode="json") if hasattr(sel, "model_dump") else dict(sel or {})

        # ------------------ EARLY EXIT: terminal no-tool ------------------
        # If the selector says there is NO suitable tool (choices empty) and
        # it's not asking a clarification question, and it gives a reason or
        # explanation, we return this result AS-IS (no top-up!).
        conv = sel_json.get("conversation") or {}
        status = str(conv.get("status", "complete")).lower()
        has_choices = bool(sel_json.get("choices"))
        reason = sel_json.get("reason")
        explanation = sel_json.get("explanation")

        if (not has_choices) and (status != "needs_clarification") and (reason or (explanation and str(explanation).strip())):
            # Normalize shape for downstream code
            sel_json["conversation"] = {"status": "complete"}
            # If enums were serialized as objects, convert to string
            if hasattr(reason, "value"):
                sel_json["reason"] = reason.value
            return sel_json
        # ------------------------------------------------------------------

        selected_names = [c.get("name") for c in sel_json.get("choices", []) if c.get("name")]
        selected_names = [n for i, n in enumerate(selected_names) if n not in selected_names[:i]]

        if len(selected_names) < num_choices:
            for h in hits:
                nm = h["doc"].name
                if nm not in selected_names:
                    selected_names.append(nm)
                if len(selected_names) >= num_choices:
                    break

            new_choices = []
            for i, nm in enumerate(selected_names[:num_choices], start=1):
                existing = next((c for c in sel_json.get("choices", []) if c.get("name") == nm), None)
                if existing:
                    c = dict(existing)
                    c["rank"] = i
                else:
                    rr = float(next((x.get("rerank_score") or x.get("__rerank__") or 0.0 for x in hits if x["doc"].name == nm), 0.0))
                    sim = float(next((x.get("score") or x.get("__sim__") or 0.0 for x in hits if x["doc"].name == nm), 0.0))
                    base = 85.0 if i == 1 else 75.0
                    acc = max(60.0, min(98.0, base + rr * 10.0))
                    c = {
                        "name": nm,
                        "rank": i,
                        "accuracy": float(acc),
                        "why": f"High retrieval/reranker match (rerank={rr:.3f}, sim={sim:.3f}).",
                    }
                new_choices.append(c)
            sel_json["choices"] = new_choices

        sel_json["choices"] = sel_json.get("choices", [])[:num_choices]
        return sel_json
