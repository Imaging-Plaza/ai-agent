# retriever/embedders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

import json, hashlib
from pathlib import Path

# ----------------------------
# Data model
# ----------------------------

class SoftwareDoc(BaseModel):
    """
    Minimal software doc used for retrieval and ranking.
    Tolerant to catalog variation (extra='ignore') and normalizes common types.
    Also derives dims/anatomy/modality from supportingData if missing.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Identity
    name: str
    url: Optional[str] = None
    repo_url: Optional[str] = None
    description: Optional[str] = None

    # Semantics
    category: List[str] = Field(default_factory=list, alias="applicationCategory")
    tasks: List[str] = Field(default_factory=list, alias="featureList")
    modality: List[str] = Field(default_factory=list, alias="imagingModality")
    keywords: List[str] = Field(default_factory=list)

    # Anatomy / dims
    dims: List[int] = Field(default_factory=list)     # derived from supportingData[*].hasDimensionality when absent
    anatomy: List[str] = Field(default_factory=list)  # derived from supportingData[*].bodySite when absent

    # Tech details
    programming_language: Optional[str] = Field(default=None, alias="programmingLanguage")
    software_requirements: List[str] = Field(default_factory=list, alias="softwareRequirements")
    gpu_required: Optional[bool] = Field(default=None, alias="requiresGPU")
    is_free: Optional[bool] = Field(default=None, alias="isAccessibleForFree")
    is_based_on: List[str] = Field(default_factory=list, alias="isBasedOn")
    plugin_of: List[str] = Field(default_factory=list, alias="isPluginModuleOf")
    related_organizations: List[str] = Field(default_factory=list, alias="relatedToOrganization")
    license: Optional[str] = None

    # Misc
    os: List[str] = Field(default_factory=list)
    weights_available: Optional[bool] = None

    # Demo / Spaces info
    runnable_example: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list, alias="runnableExample"
    )
    has_executable_notebook: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list, alias="hasExecutableNotebook"
    )

    @field_validator("runnable_example", "has_executable_notebook", mode="before")
    @classmethod
    def _coerce_list_any(cls, v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    # --------- Derive dims/anatomy/modality from supportingData BEFORE field validators ---------
    @model_validator(mode="before")
    @classmethod
    def _derive_from_supporting_data(cls, data: Any):
        if not isinstance(data, dict):
            return data

        sd = data.get("supportingData")
        if not sd:
            return data

        # normalize to list of dicts
        items = sd if isinstance(sd, list) else [sd]

        # collect from nested records
        dims_collected: List[int] = []
        anatomy_collected: List[str] = []
        mod_extra: List[str] = []

        def push_dim(x):
            try:
                xi = int(x)
                if xi not in dims_collected:
                    dims_collected.append(xi)
            except Exception:
                # tolerate "3D"/"2-D"/"volumetric" etc.
                if isinstance(x, str):
                    s = x.strip().lower().replace(" ", "")
                    if s in {"2", "2d", "2-d"}:
                        if 2 not in dims_collected: dims_collected.append(2); return
                    if s in {"3", "3d", "3-d", "volume", "volumetric", "stack"}:
                        if 3 not in dims_collected: dims_collected.append(3); return
                    if s in {"4", "4d", "4-d", "timeseries", "time-series", "temporal"}:
                        if 4 not in dims_collected: dims_collected.append(4); return
                    digits = "".join(ch for ch in s if ch.isdigit())
                    if digits:
                        try:
                            xi = int(digits)
                            if xi not in dims_collected:
                                dims_collected.append(xi)
                        except Exception:
                            pass

        for it in items:
            if not isinstance(it, dict):
                continue

            # hasDimensionality
            hd = it.get("hasDimensionality")
            if hd is not None:
                if isinstance(hd, list):
                    for v in hd: push_dim(v)
                else:
                    push_dim(hd)

            # bodySite
            bs = it.get("bodySite")
            if bs is not None:
                vals = bs if isinstance(bs, list) else [bs]
                for v in vals:
                    s = str(v).strip()
                    if s and s not in anatomy_collected:
                        anatomy_collected.append(s)

            # imagingModality (nested)
            im = it.get("imagingModality")
            if im is not None:
                vals = im if isinstance(im, list) else [im]
                for v in vals:
                    s = str(v).strip()
                    if s and s not in mod_extra:
                        mod_extra.append(s)

        # populate only if missing/empty at top-level
        if not data.get("dims") and dims_collected:
            data["dims"] = dims_collected
        if not data.get("anatomy") and anatomy_collected:
            data["anatomy"] = anatomy_collected
        # merge nested imagingModality into top-level if present
        if mod_extra:
            top = data.get("imagingModality") or data.get("modality") or []
            top_list = top if isinstance(top, list) else [top]
            merged = []
            for v in top_list + mod_extra:
                s = str(v).strip()
                if s and s not in merged:
                    merged.append(s)
            data["imagingModality"] = merged

        return data

    # ---------- validators / coercers ----------
    @staticmethod
    def _as_list_of_str(v) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            out = []
            for x in v:
                if x is None:
                    continue
                s = str(x).strip()
                if s and s not in out:
                    out.append(s)
            return out
        s = str(v).strip()
        return [s] if s else []

    @field_validator(
        "category", "tasks", "modality", "keywords",
        "software_requirements", "is_based_on", "related_organizations",
        "os", "plugin_of",
        mode="before"
    )
    @classmethod
    def _coerce_list_strs(cls, v):
        return cls._as_list_of_str(v)

    @field_validator("programming_language", "license", mode="before")
    @classmethod
    def _coerce_scalar_from_list(cls, v):
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    return x
            return None
        return v

    @field_validator("gpu_required", "is_free", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes", "y", "on"}:
                return True
            if s in {"false", "0", "no", "n", "off"}:
                return False
        return None

    @field_validator("dims", mode="before")
    @classmethod
    def _coerce_dims(cls, v):
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        out: List[int] = []

        def push(x):
            try:
                xi = int(x)
                if xi not in out:
                    out.append(xi)
            except Exception:
                pass

        for it in items:
            if isinstance(it, (int, float)):
                push(it); continue
            if not isinstance(it, str):
                continue
            s = it.strip().lower().replace(" ", "")
            if s in {"2", "2d", "2-d"}: push(2); continue
            if s in {"3", "3d", "3-d", "volume", "volumetric", "stack"}: push(3); continue
            if s in {"4", "4d", "4-d", "timeseries", "time-series", "temporal"}: push(4); continue
            digits = "".join(ch for ch in s if ch.isdigit())
            if digits: push(digits)
        return out

    def to_retrieval_text(self) -> str:
        dims_str = ", ".join(f"{d}D" for d in (self.dims or []))
        parts = [
            f"name: {self.name}",
            f"tasks: {', '.join(self.tasks)}" if self.tasks else "",
            f"modality: {', '.join(self.modality)}" if self.modality else "",
            f"dims: {dims_str}" if dims_str else "",
            f"category: {', '.join(self.category)}" if self.category else "",
            f"keywords: {', '.join(self.keywords)}" if self.keywords else "",
            f"language: {self.programming_language or ''}",
            f"license: {self.license or ''}",
            f"gpu_required: {self.gpu_required}",
            f"is_free: {self.is_free}",
            f"plugin_of: {', '.join(self.plugin_of)}" if self.plugin_of else "",
            f"based_on: {', '.join(self.is_based_on)}" if self.is_based_on else "",
            f"orgs: {', '.join(self.related_organizations)}" if self.related_organizations else "",
            f"desc: {self.description or ''}",
        ]
        return " | ".join(p for p in parts if p)


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

def _fingerprint_doc(doc: SoftwareDoc) -> str:
    # Stable content hash; ignore volatile fields if you have any
    payload = doc.model_dump(mode="json", exclude_none=True)
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

class VectorIndex:
    """
    Cosine-similarity FAISS index (inner product on normalized vectors).
    - Uses IndexIDMap2 to support deletes/updates by id.
    - Persists with faiss.write_index/read_index + a JSON sidecar.
    - Can sync against a changing catalog efficiently.
    """
    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder

        # Use ID map so we can remove/update by FAISS int64 ids.
        base = faiss.IndexFlatIP(embedder.dim)
        self._index = faiss.IndexIDMap2(base)

        # Metadata
        self.id_to_faiss: Dict[str, int] = {}           # string id -> int64
        self.faiss_to_id: Dict[int, str] = {}           # int64 -> string id
        self.docs: Dict[str, SoftwareDoc] = {}          # string id -> SoftwareDoc
        self.fingerprints: Dict[str, str] = {}          # string id -> hash
        self._next_faiss_id: int = 1                    # simple monotonically increasing int64
                                                        # (avoid 0, reserved in some contexts)

    # ---------- Core ops ----------

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
        fids = [self.id_to_faiss[sid] for sid in sids if sid in self.id_to_faiss]
        if not fids:
            return
        arr = np.array(fids, dtype=np.int64)
        self._index.remove_ids(arr)
        # purge metadata
        for sid, fid in zip(sids, fids):
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
        embs = self.embedder.embed_corpus(texts)  # normalized float32

        fids = []
        for it in items:
            # if already exists, remove first
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
        reranker: Optional[CrossEncoderReranker] = None,
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
            reranked = reranker.rerank(query_text, texts, top_k=min(rerank_top_k, len(hits)))
            return [hits[i] | {"rerank_score": s} for i, s in reranked]
        return hits

    # ---------- Catalog sync ----------

    def sync_with_catalog(self, items: List[IndexItem]) -> Dict[str, int]:
        """
        Diff the incoming catalog against what's stored, re-embed only what's changed,
        remove what's gone. Returns counts: {'added':..., 'updated':..., 'removed':...}
        """
        incoming_by_id = {it.id: it for it in items}
        incoming_ids = set(incoming_by_id.keys())
        current_ids = set(self.docs.keys())

        # Deletions
        to_remove = sorted(current_ids - incoming_ids)

        # Updates / Adds (by fingerprint)
        to_add, to_update = [], []
        for it in items:
            fp = _fingerprint_doc(it.doc)
            prev = self.fingerprints.get(it.id)
            if prev is None:
                to_add.append(it)
            elif prev != fp:
                to_update.append(it)

        # Apply changes
        self._remove_by_ids(to_remove)
        if to_add:
            self.upsert(to_add)
        if to_update:
            self.upsert(to_update)

        return {"added": len(to_add), "updated": len(to_update), "removed": len(to_remove)}

    # ---------- Persistence ----------

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
                # store the model name if available (SentenceTransformer)
                "model_name": getattr(getattr(self.embedder, "model", None), "model_card", None)
                               or getattr(self.embedder, "model_name", None)
                               or "unknown",
                "dim": self.embedder.dim,
                "query_prefix": getattr(self.embedder, "query_prefix", ""),
                "doc_prefix": getattr(self.embedder, "doc_prefix", ""),
                "normalized": True,
                "metric": "ip",
            },
            "next_faiss_id": self._next_faiss_id,
            "id_to_faiss": self.id_to_faiss,                           # str -> int
            "docs": {sid: self.docs[sid].model_dump(mode="json", exclude_none=True)
                     for sid in self.docs},
            "fingerprints": self.fingerprints,
        }
        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def load(cls, dirpath: str | Path, embedder: TextEmbedder) -> "VectorIndex":
        p = Path(dirpath)
        faiss_path = p / "index.faiss"
        meta_path  = p / "meta.json"

        # Explicitly check first
        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing index artifacts in {p} "
                                    f"(expected {faiss_path.name} and {meta_path.name})")

        idx = cls(embedder)
        try:
            idx._index = faiss.read_index(str(faiss_path))
        except Exception as e:
            # faiss throws RuntimeError with "could not open ..." message
            msg = str(e).lower()
            if "could not open" in msg or "no such file" in msg:
                raise FileNotFoundError(str(faiss_path)) from e
            raise

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("embedder", {}).get("dim") != embedder.dim:
            raise ValueError(
                f"Embedder dim mismatch: saved={meta.get('embedder', {}).get('dim')} vs current={embedder.dim}"
            )

        idx._next_faiss_id = int(meta.get("next_faiss_id", 1))
        idx.id_to_faiss = {str(k): int(v) for k, v in meta.get("id_to_faiss", {}).items()}
        idx.faiss_to_id = {int(v): str(k) for k, v in idx.id_to_faiss.items()}
        idx.docs = {sid: SoftwareDoc(**payload) for sid, payload in meta.get("docs", {}).items()}
        idx.fingerprints = {str(k): str(v) for k, v in meta.get("fingerprints", {}).items()}
        return idx
