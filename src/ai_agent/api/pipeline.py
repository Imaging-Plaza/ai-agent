# api/pipeline.py
from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from retriever.embedders import (
    LocalBGEEmbedder,
    VectorIndex,
    IndexItem,
    CrossEncoderReranker,
    SoftwareDoc,  # ensure SoftwareDoc maps your catalog; optional .runnables supported
)
from generator.generator import VLMToolSelector
from generator.schema import CandidateDoc, NoToolReason  # Add NoToolReason to the import

# NOTE: these now handle LISTS of paths and richer metadata
from utils.image_meta import summarize_image_metadata, detect_ext_token
# New helpers for loading and previews
from utils.image_io import load_any
from utils.previews import mip_montage, slice_gif, stack_sweep_gif, contact_sheet_slices

log = logging.getLogger("pipeline")


class RAGImagingPipeline:
    """
    Retrieval (text-only) + single-call VLM selection (multi-image aware: image(s)/volume(s) + text + candidates + metadata)
    + base runnable link. Persists FAISS to disk and keeps it in sync with catalog changes.
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

    # -------- Retrieval (text-only, optional format token from MULTIPLE paths) --------
    def recommend(
        self, user_task: str, image_paths: Optional[List[str]], top_k: int = 5
    ) -> Tuple[List[dict], Dict[str, float]]:
        """
        image_paths: list of file/folder/zip paths. We don't load data here; we only derive a format token.
        """
        ext_tok = detect_ext_token(image_paths)  # e.g., "DICOM NIfTI TIFF"
        query = (user_task or "").strip()
        if ext_tok:
            query = f"{query} format:{ext_tok}"

        raw_hits = self.index.search(query, k=50, reranker=None)
        hits = self._apply_reranker(query, raw_hits, top_k=top_k)

        # score summary
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

    # -------- Build a single preview (PNG/GIF) + metadata text for VLM --------
    def _build_preview_for_vlm(self, image_paths: Optional[List[str]]) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (preview_path, meta_text). preview_path is a PNG (MIP montage) or GIF (slice sweep),
        stored in a temp dir. meta_text is a concise summary of all inputs.
        """
        if not image_paths:
            return None, None

        meta_text = None
        try:
            meta_text = summarize_image_metadata(image_paths)
        except Exception:
            log.exception("Image metadata summarization failed; continuing without metadata.")

        # choose the first readable item and create a compact visual summary
        for p in image_paths:
            try:
                data, meta = load_any(p)
                shp = getattr(meta, "shape", None) or meta.get("shape")
                if shp is None:
                    shp = getattr(data, "shape", None)
                if shp is None:
                    continue

                tmpdir = Path(tempfile.mkdtemp(prefix="preview_"))

                # 3D volume -> MIP montage
                if len(shp) == 3:
                    # 3D: create BOTH a contact-sheet PNG (for the VLM) and a GIF (for humans)
                    png_path = tmpdir / "slices_grid.png"
                    gif_path = tmpdir / "sweep.gif"

                    # 1) Contact-sheet PNG (preferred for VLM)
                    try:
                        contact_sheet_slices(data, png_path, max_slices=36, grid_cols=6)
                    except Exception:
                        # fallback to MIP if grid fails
                        try:
                            mip_montage(data, png_path)
                        except Exception:
                            log.warning("Failed to create MIP/ContactSheet PNG; will try GIF only.")

                    # 2) Optional GIF for debugging (not passed to VLM)
                    try:
                        stack_sweep_gif(data, gif_path, fps=12, max_frames=64)
                        log.info("Wrote 3D GIF preview: %s (%d bytes)", gif_path, os.path.getsize(gif_path))
                    except Exception:
                        pass

                    if png_path.exists():
                        log.info("Using PNG preview for VLM: %s (%d bytes)", png_path, os.path.getsize(png_path))
                        return str(png_path), meta_text

                    # As an absolute fallback, if PNG failed but GIF exists (selector may convert it)
                    if gif_path.exists():
                        log.info("Falling back to GIF for VLM: %s", gif_path)
                        return str(gif_path), meta_text

                # 4D -> mean over time, slice sweep GIF
                if len(shp) == 4:
                    vol = np.asarray(data).mean(axis=-1)
                    out = tmpdir / "sweep.gif"
                    step = max(1, vol.shape[2] // 64)
                    slice_gif(vol, out, axis=2, step=step, fps=12)
                    return str(out), meta_text

                # 2D image -> save as PNG
                if len(shp) == 2:
                    out = tmpdir / "image.png"
                    try:
                        import imageio.v3 as iio
                        arr = data
                        if arr.dtype != np.uint8:
                            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                        iio.imwrite(str(out), arr)
                        return str(out), meta_text
                    except Exception:
                        # if writing fails, skip to next path
                        pass
            except Exception:
                continue

        return None, meta_text

    # -------- Selection (0-call if confident, else 1-call VLM with preview) --------
    def _select(
        self, user_task: str, hits: List[dict], image_paths: Optional[List[str]]
    ) -> Tuple[Optional[dict], Dict[str, Any]]:
        # Build candidates using TOP_K from environment
        top_k = int(os.getenv("TOP_K", "8"))
        candidates = []
        for h in hits[:top_k]:  # Use TOP_K instead of hardcoded 5
            try:
                candidates.append(CandidateDoc(**h["doc"].model_dump()))
            except Exception:
                continue

        # Quick exit if retrieval already looks confident
        if self._is_confident(hits):
            chosen = hits[0]
            sel_json = {
                "choices": [{
                    "name": chosen["doc"].name,
                    "rank": 1,
                    "accuracy": 100.0,  # High confidence
                    "why": "High-confidence retrieval; top candidate clearly dominates."
                }],
            }
            return chosen, sel_json

        log.info("Using VLM selector (FORCE_VLM=%s)", getattr(self, "force_vlm", None))

        preview_path, meta_text = self._build_preview_for_vlm(image_paths)
        log.info("Selector preview → %s", preview_path or "<none>")

        try:
            sel = self.selector_vlm.select(
                user_task=user_task,
                candidates=candidates,
                image_path=preview_path,
                image_meta=meta_text,
            )
            sel_json = sel.model_dump()

            # Handle no suitable tools case
            if not sel.choices:
                return None, sel_json

            # Map to hit or fallback
            chosen = next((h for h in hits if h["doc"].name == sel.choices[0].name), None)
            if chosen is None and hits:
                chosen = hits[0]
            return chosen, sel_json

        except Exception as e:
            log.exception("VLM selector failed. Falling back to top-1.")
            if not hits:
                return None, {
                    "choices": [],
                    "reason": NoToolReason.NO_MATCHES
                }

            chosen = hits[0]
            return chosen, {
                "choices": [{
                    "name": chosen["doc"].name,
                    "rank": 1,
                    "accuracy": 50.0,
                    "why": f"Fallback to retrieval due to error: {str(e)}"
                }],
                "reason": NoToolReason.FALLBACK_TO_RETRIEVAL
            }

    # -------- Confidence gate --------
    def _is_confident(self, hits: List[dict]) -> bool:
        if self.force_vlm or not hits:
            return False
        top = float(hits[0].get("rerank_score") or hits[0].get("score", 0.0))
        second = float(hits[1].get("rerank_score") or hits[1].get("score", 0.0)) if len(hits) > 1 else 0.0
        return (top - second) > self.conf_margin or top >= self.conf_top

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
        for items in (getattr(doc, "runnable_example", None) or [], getattr(doc, "has_executable_notebook", None) or []):
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
    def recommend_and_link(self, image_paths: Optional[List[str]], user_task: str) -> Dict[str, Any]:
        """
        image_paths: list of file/folder/zip paths; can include DICOM series (dirs/zips), NIfTI, TIFF stacks, etc.
        Returns multiple ranked choices with explanations.
        """
        top_k = int(os.getenv("TOP_K", "8"))
        num_choices = int(os.getenv("NUM_CHOICES", "3"))
        hits, scores = self.recommend(user_task, image_paths, top_k=top_k)
        if not hits:
            return {"error": "No candidates found."}

        chosen, selection = self._select(user_task, hits, image_paths)
        
        # Convert selection to new format with multiple ranked choices
        if isinstance(selection.get("choices"), list):
            # Already in new format
            result = {
                "choices": selection["choices"],
                "scores": {k: round(v, 3) for k, v in scores.items()},
            }
        else:
            # Convert old format to new
            choices = []
            if chosen:
                choices.append({
                    "name": chosen["doc"].name,
                    "rank": 1,
                    "why": selection.get("why", "")
                })
                # Add alternates as lower-ranked choices
                for i, alt in enumerate(selection.get("alternates", [])[:num_choices-1], 2):
                    choices.append({
                        "name": alt,
                        "rank": i,
                        "why": f"Alternative choice #{i}"
                    })
            
            result = {
                "choices": choices,
                "scores": {k: round(v, 3) for k, v in scores.items()},
            }

        # Add demo links for each choice
        for choice in result["choices"]:
            doc = next((h["doc"] for h in hits if h["doc"].name == choice["name"]), None)
            if doc:
                link = self._best_runnable_link(doc)
                if link:
                    choice["demo_link"] = link

        return result