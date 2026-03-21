from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import faiss
import numpy as np

from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.retriever.text_embedder import TextEmbedder

if TYPE_CHECKING:
    from .reranker import CrossEncoderReranker


log = logging.getLogger("retriever.vector_index")


@dataclass
class IndexItem:
    id: str
    doc: SoftwareDoc


def _fingerprint_doc(doc: SoftwareDoc) -> str:
    def _sorted_unique(xs):
        return sorted({str(x).strip() for x in (xs or []) if str(x).strip()})

    payload = {
        "name": (doc.name or "").strip(),
        "description": (doc.description or "").strip(),
        "category": _sorted_unique(doc.category),
        "tasks": _sorted_unique(doc.tasks),
        "modality": _sorted_unique(doc.modality),
        "keywords": _sorted_unique(doc.keywords),
        "dims": sorted(set(doc.dims or [])),
        "anatomy": _sorted_unique(doc.anatomy),
        "programming_language": (doc.programming_language or "").strip(),
        "license": (doc.license or "").strip(),
        "software_requirements": _sorted_unique(doc.software_requirements),
        "gpu_required": (
            bool(doc.gpu_required) if doc.gpu_required is not None else None
        ),
        "is_free": bool(doc.is_free) if doc.is_free is not None else None,
        "is_based_on": _sorted_unique(doc.is_based_on),
        "plugin_of": _sorted_unique(doc.plugin_of),
        "related_organizations": _sorted_unique(doc.related_organizations),
        "os": _sorted_unique(doc.os),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class VectorIndex:
    """
    Cosine-similarity FAISS index (inner product on normalized vectors).
    - Uses IndexIDMap2 to support deletes/updates by id.
    - Persists with faiss.write_index/read_index + a JSON sidecar.
    - Can sync against a changing catalog efficiently.
    """

    FINGERPRINT_VERSION = 2

    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder
        base = faiss.IndexFlatIP(embedder.dim)
        self._index = faiss.IndexIDMap2(base)
        self.id_to_faiss: Dict[str, int] = {}
        self.faiss_to_id: Dict[int, str] = {}
        self.docs: Dict[str, SoftwareDoc] = {}
        self.fingerprints: Dict[str, str] = {}
        self._next_faiss_id: int = 1

    def _assign_faiss_id(self, sid: str) -> int:
        if sid in self.id_to_faiss:
            return self.id_to_faiss[sid]
        fid = self._next_faiss_id
        self._next_faiss_id += 1
        self.id_to_faiss[sid] = fid
        self.faiss_to_id[fid] = sid
        return fid

    def _remove_by_ids(self, sids: List[str]) -> None:
        if not sids:
            return
        present_sids = [sid for sid in sids if sid in self.id_to_faiss]
        if not present_sids:
            return
        fids = [self.id_to_faiss[sid] for sid in present_sids]
        arr = np.array(fids, dtype=np.int64)
        self._index.remove_ids(arr)
        for sid, fid in zip(present_sids, fids):
            self.id_to_faiss.pop(sid, None)
            self.faiss_to_id.pop(fid, None)
            self.docs.pop(sid, None)
            self.fingerprints.pop(sid, None)

    def upsert(self, items: List[IndexItem]) -> None:
        """
        Upsert without diffing (assumes the caller knows what changed).
        For general catalog changes prefer sync_with_catalog.
        """

        if not items:
            return
        texts = [it.doc.to_retrieval_text() for it in items]
        embs = self.embedder.embed_corpus(texts)
        fids = []
        for it in items:
            if it.id in self.id_to_faiss:
                self._remove_by_ids([it.id])
            fid = self._assign_faiss_id(it.id)
            fids.append(fid)
        self._index.add_with_ids(embs, np.array(fids, dtype=np.int64))
        for it, fid in zip(items, fids):
            self.docs[it.id] = it.doc
            self.fingerprints[it.id] = _fingerprint_doc(it.doc)

    def search(
        self,
        query_text: str,
        k: int = 20,
        reranker: Optional["CrossEncoderReranker"] = None,
        rerank_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        q = self.embedder.embed_queries([query_text])
        total = self._index.ntotal
        if total == 0:
            return []
        k = min(k, total)
        D, I = self._index.search(q, k)
        scores = D[0].tolist()
        fids = I[0].tolist()
        hits = []
        for score, fid in zip(scores, fids):
            if fid == -1:
                continue
            sid = self.faiss_to_id.get(int(fid))
            if not sid:
                continue
            hits.append({"id": sid, "doc": self.docs[sid], "score": float(score)})
        if reranker and hits:
            texts = [h["doc"].to_retrieval_text() for h in hits]
            reranked = reranker.rerank(
                query_text, texts, top_k=min(rerank_top_k, len(hits))
            )
            return [hits[i] | {"rerank_score": s} for i, s in reranked]
        return hits

    def sync_with_catalog(self, items: List[IndexItem]) -> Dict[str, int]:
        """
        Diff the incoming catalog against what's stored, re-embed only what's changed,
        remove what's gone. Returns counts: {'added':..., 'updated':..., 'removed':...}
        """

        incoming_by_id = {it.id: it for it in items}
        incoming_ids = set(incoming_by_id.keys())
        current_ids = set(self.docs.keys())
        to_remove = sorted(current_ids - incoming_ids)
        to_add, to_update = [], []
        for it in items:
            fp = _fingerprint_doc(it.doc)
            prev = self.fingerprints.get(it.id)
            if prev is None:
                to_add.append(it)
            elif prev != fp:
                to_update.append(it)
        self._remove_by_ids(to_remove)
        if to_add:
            self.upsert(to_add)
        if to_update:
            self.upsert(to_update)
        added_n, updated_n, removed_n = len(to_add), len(to_update), len(to_remove)
        if added_n or updated_n or removed_n:

            def sample_ids(seq, n: int = 5):
                if not seq:
                    return []
                if isinstance(seq[0], IndexItem):
                    return [it.id for it in seq[:n]]
                return [str(x) for x in seq[:n]]

            log.info(
                "Catalog changed: added=%d, updated=%d, removed=%d",
                added_n,
                updated_n,
                removed_n,
            )
            add_sample = sample_ids(to_add)
            upd_sample = sample_ids(to_update)
            rem_sample = sample_ids(to_remove)
            if add_sample:
                log.info(
                    "  added ids (sample): %s%s",
                    ", ".join(add_sample),
                    " ..." if added_n > len(add_sample) else "",
                )
            if upd_sample:
                log.info(
                    "  updated ids (sample): %s%s",
                    ", ".join(upd_sample),
                    " ..." if updated_n > len(upd_sample) else "",
                )
            if rem_sample:
                log.info(
                    "  removed ids (sample): %s%s",
                    ", ".join(rem_sample),
                    " ..." if removed_n > len(rem_sample) else "",
                )

        return {"added": added_n, "updated": updated_n, "removed": removed_n}

    def save(self, dirpath: str | Path) -> None:
        """
        Save FAISS index + sidecar metadata. Fast reload via read_index.
        """

        p = Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "index.faiss"))
        meta = {
            "version": 1,
            "embedder": {
                "type": self.embedder.__class__.__name__,
                "model_name": getattr(
                    getattr(self.embedder, "model", None), "model_card", None
                )
                or getattr(self.embedder, "model_name", None)
                or "unknown",
                "dim": self.embedder.dim,
                "query_prefix": getattr(self.embedder, "query_prefix", ""),
                "doc_prefix": getattr(self.embedder, "doc_prefix", ""),
                "normalized": True,
                "metric": "ip",
            },
            "next_faiss_id": self._next_faiss_id,
            "id_to_faiss": self.id_to_faiss,
            "docs": {
                sid: self.docs[sid].model_dump(mode="json", exclude_none=True)
                for sid in self.docs
            },
            "fingerprints": self.fingerprints,
            "fingerprint_version": self.FINGERPRINT_VERSION,
        }
        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                meta, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )

    @classmethod
    def load(cls, dirpath: str | Path, embedder: TextEmbedder) -> "VectorIndex":
        p = Path(dirpath)
        faiss_path = p / "index.faiss"
        meta_path = p / "meta.json"
        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing index artifacts in {p} (expected {faiss_path.name} and {meta_path.name})"
            )
        idx = cls(embedder)
        try:
            idx._index = faiss.read_index(str(faiss_path))
        except Exception as e:
            msg = str(e).lower()
            if "could not open" in msg or "no such file" in msg:
                raise FileNotFoundError(str(faiss_path)) from e
            raise
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        expected = cls.FINGERPRINT_VERSION
        found = int(meta.get("fingerprint_version", 0))
        if found != expected:
            idx.fingerprints = {}
        else:
            idx.fingerprints = {
                str(k): str(v) for k, v in meta.get("fingerprints", {}).items()
            }

        if meta.get("embedder", {}).get("dim") != embedder.dim:
            raise ValueError(
                f"Embedder dim mismatch: saved={meta.get('embedder', {}).get('dim')} vs current={embedder.dim}"
            )
        idx._next_faiss_id = int(meta.get("next_faiss_id", 1))
        idx.id_to_faiss = {
            str(k): int(v) for k, v in meta.get("id_to_faiss", {}).items()
        }
        idx.faiss_to_id = {int(v): str(k) for k, v in idx.id_to_faiss.items()}
        idx.docs = {
            sid: SoftwareDoc(**payload) for sid, payload in meta.get("docs", {}).items()
        }

        return idx
