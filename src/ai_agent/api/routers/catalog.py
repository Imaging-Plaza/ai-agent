"""Return the FAISS catalog as a list of compact CatalogEntry rows."""

from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, Depends

from ai_agent.api.deps import get_doc_index, require_auth
from ai_agent.api.schemas import CatalogEntry
from ai_agent.retriever.software_doc import SoftwareDoc

router = APIRouter(
    prefix="/api/catalog", tags=["catalog"], dependencies=[Depends(require_auth)]
)


@router.get("", response_model=List[CatalogEntry])
def list_catalog(
    doc_index: Dict[str, SoftwareDoc] = Depends(get_doc_index),
) -> List[CatalogEntry]:
    out: List[CatalogEntry] = []
    for name, doc in sorted(doc_index.items(), key=lambda kv: kv[0].lower()):
        out.append(
            CatalogEntry(
                name=name,
                description=getattr(doc, "description", "") or "",
                modality=list(getattr(doc, "modality", []) or []),
                license=getattr(doc, "license", None),
                dims=list(getattr(doc, "dims", []) or []),
                keywords=list(getattr(doc, "keywords", []) or [])[:20],
            )
        )
    return out
