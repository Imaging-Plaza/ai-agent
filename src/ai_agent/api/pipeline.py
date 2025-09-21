# api/pipeline.py
from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import imageio.v3 as iio

from retriever.embedders import (
    LocalBGEEmbedder,
    VectorIndex,
    IndexItem,
    CrossEncoderReranker,
    SoftwareDoc,  # ensure SoftwareDoc maps your catalog; optional .runnables supported
)
from generator.generator import VLMToolSelector
from generator.schema import CandidateDoc, NoToolReason, ConversationStatus, ToolSelection  # Add NoToolReason to the import
from utils.file_validator import FileValidator

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
    def _select(self, hits, image_meta_text, user_task, preview_path):
        """
        Always run the selector (no short-circuit). Ask for up to NUM_CHOICES.
        Fallback: if the selector returns < NUM_CHOICES, fill from remaining hits.
        """
        num_choices = int(os.getenv("NUM_CHOICES", "3"))

        # Build CandidateDoc list from retrieval hits
        candidates = []
        for h in hits:
            d = h["doc"]            # this is your SoftwareDoc (pydantic)
            # Convert SoftwareDoc -> CandidateDoc (let pydantic handle aliases)
            try:
                cd = CandidateDoc.model_validate(d.model_dump(mode="python"))
            except Exception:
                # last-resort loose mapping
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

        # Call the VLM selector with the proper signature
        sel = self.selector_vlm.select(
            user_task=user_task,
            candidates=candidates,
            image_path=preview_path,          # <- IMPORTANT
            image_meta=image_meta_text or "",
        )

        # Normalize to dict
        sel_json = sel.model_dump() if hasattr(sel, "model_dump") else dict(sel or {})

        # Ensure up to NUM_CHOICES by topping up with remaining hits (preserve order)
        selected_names = [c.get("name") for c in sel_json.get("choices", []) if c.get("name")]
        selected_names = [n for i, n in enumerate(selected_names) if n not in selected_names[:i]]

        if len(selected_names) < num_choices:
            for h in hits:
                nm = h["doc"].name
                if nm not in selected_names:
                    selected_names.append(nm)
                if len(selected_names) >= num_choices:
                    break

            # rebuild choices with rank; keep model’s fields if present
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

        # Cap (in case the model gave too many)
        sel_json["choices"] = sel_json.get("choices", [])[:num_choices]
        return sel_json



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
    def recommend_and_link(self, image_paths: Optional[List[str]], user_task: str, 
                       conversation_history: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process query and return recommendations (no short-circuit; always up to NUM_CHOICES)."""

        # ------------- Build enriched task text (include prior turns) -------------
        full_task = user_task
        if conversation_history:
            full_task = "\\n".join([
                "Previous conversation:",
                *conversation_history,
                "\\nCurrent request:",
                user_task
            ])

        # ------------- Validate files --------------------------------------------
        if image_paths:
            valid_paths, errors = FileValidator.validate_files(image_paths)
            if errors:
                return {
                    "error": "File validation failed:\\n" + "\\n".join(errors),
                    "choices": [],
                    "reason": NoToolReason.INVALID_FILES
                }
            image_paths = valid_paths

        # ------------- Optional: extract preview/meta for better prompting --------
        preview_path = None
        image_meta_text = ""
        try:
            preview_path, image_meta_text = self._build_preview_for_vlm(image_paths or [])
        except Exception:
            image_meta_text = ""

        # ------------- Retrieve candidates ---------------------------------------
        top_k = int(os.getenv("TOP_K", "8"))
        num_choices = int(os.getenv("NUM_CHOICES", "3"))
        hits, scores = self.recommend(full_task, image_paths, top_k=top_k)
        if not hits:
            return {"error": "No candidates found."}

        # Build compact candidate hints to condition the selector
        # (name | tasks | modality | dims | lang)
        cand_lines = []
        for h in hits:
            d = h["doc"]
            cand_lines.append(
                f"- {getattr(d, 'name', '?')} | "
                f"tasks={','.join(getattr(d, 'tasks', []) or [])} | "
                f"modality={','.join(getattr(d, 'modality', []) or [])} | "
                f"dims={','.join(map(str, getattr(d, 'dims', []) or []))} | "
                f"lang={getattr(d, 'lang', '')}"
            )

        selector_task = full_task
        if image_meta_text:
            selector_task += f"\\n\\nImage metadata: {image_meta_text}"
        if cand_lines:
            selector_task += "\\n\\nCandidate tool hints:\\n" + "\\n".join(cand_lines)
        selector_task += f"\\n\\nRequest up to {num_choices} choices."

        # ------------- Run selector (no short-circuit here) -----------------------
        # FIX: pass arguments in the correct order, including conversation history
        selection = self._select(hits, image_meta_text, selector_task, preview_path)

        # Convert to conversation format
        result = {
            "conversation": selection.get("conversation", {"status": "complete"}),
            "choices": selection.get("choices", []),
        }

        # ------------- Enforce NUM_CHOICES & add demo links -----------------------
        def _fallback_score(i: int, hit: dict) -> float:
            """Simple bounded fallback accuracy if selector didn't provide one."""
            rr = float(hit.get("rerank_score") or hit.get("__rerank__") or 0.0)
            base = 85.0 if i == 1 else 75.0
            return max(60.0, min(98.0, base + rr * 10.0))

        # If selector returned < num_choices, top up from remaining hits
        chosen_names = [c.get("name") for c in result["choices"] if c.get("name")]
        # de-dup preserve order
        chosen_names = [n for i, n in enumerate(chosen_names) if n and n not in chosen_names[:i]]

        if len(chosen_names) < num_choices:
            for h in hits:
                nm = h["doc"].name
                if nm not in chosen_names:
                    chosen_names.append(nm)
                if len(chosen_names) >= num_choices:
                    break

            # rebuild choices keeping selector fields when present
            new_choices = []
            for i, nm in enumerate(chosen_names[:num_choices], start=1):
                existing = next((c for c in result["choices"] if c.get("name") == nm), None)
                if existing:
                    # ensure rank is consistent
                    c = dict(existing)
                    c["rank"] = i
                else:
                    hit = next((x for x in hits if x["doc"].name == nm), None)
                    acc = _fallback_score(i, hit or {})
                    sim = float((hit or {}).get("score") or (hit or {}).get("__sim__") or 0.0)
                    rr  = float((hit or {}).get("rerank_score") or (hit or {}).get("__rerank__") or 0.0)
                    c = {
                        "name": nm,
                        "rank": i,
                        "accuracy": float(acc),
                        "why": f"High retrieval/reranker match (rerank={rr:.3f}, sim={sim:.3f})."
                    }
                new_choices.append(c)
            result["choices"] = new_choices

        # Cap to num_choices (in case selector over-returned)
        result["choices"] = result["choices"][:num_choices]

        # Add demo links for all choices when conversation is complete
        if result["conversation"]["status"] == "complete":
            for choice in result["choices"]:
                doc = next((h["doc"] for h in hits if h["doc"].name == choice["name"]), None)
                if doc:
                    link = self._best_runnable_link(doc)
                    if link:
                        choice["demo_link"] = link

        return result
