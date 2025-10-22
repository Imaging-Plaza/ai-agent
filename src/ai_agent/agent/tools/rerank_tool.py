from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel
import os, re

from retriever.embedders import SoftwareDoc
from .utils import get_pipeline

class RerankInput(BaseModel):
    query: str
    candidate_names: List[str]
    top_k: int = int(os.getenv("TOP_K", "5"))

class RerankOutput(BaseModel):
    reranked: List[Dict[str, Any]]
    used_model: bool

def tool_rerank(inp: RerankInput) -> RerankOutput:
    pipe = get_pipeline()
    # reconstruct minimal hit dicts for reranker from catalog
    hits: List[Dict[str, Any]] = []
    for name in inp.candidate_names:
        doc = pipe.get_doc(name)
        if not doc:
            continue
        hits.append({"id": name, "doc": doc, "score": 0.0})
    if not hits:
        return RerankOutput(reranked=[], used_model=False)
    if getattr(pipe, "reranker", None):
        ranked = pipe.rerank_only(inp.query, hits, top_k=inp.top_k)
        out = [
            {
                "name": h["doc"].name,
                "score": float(h.get("rerank_score") or h.get("score") or 0.0),
            }
            for h in ranked
        ]
        return RerankOutput(reranked=out, used_model=True)
    # fallback lexical
    q = inp.query.lower()
    scored = []
    for h in hits:
        doc: SoftwareDoc = h["doc"]
        text = " ".join(filter(None, [doc.name, " ".join(doc.tasks), doc.description or ""])).lower()
        score = 0.0
        for tok in set(re.findall(r"[a-z0-9]+", q)):
            if tok in text:
                score += 1.0
        scored.append((doc.name, score))
    scored.sort(key=lambda x: -x[1])
    return RerankOutput(reranked=[{"name": n, "score": s} for n, s in scored[: inp.top_k]], used_model=False)