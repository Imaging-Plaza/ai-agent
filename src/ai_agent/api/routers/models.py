"""Expose the model list from config.yaml to the frontend dropdown."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from ai_agent.api.deps import require_auth
from ai_agent.api.schemas import ModelOption
from ai_agent.utils.config import get_available_models_config

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(require_auth)])


@router.get("", response_model=List[ModelOption])
def list_models() -> List[ModelOption]:
    raw = get_available_models_config()
    out: List[ModelOption] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        out.append(
            ModelOption(
                display_name=m.get("display_name") or m.get("name") or "(unnamed)",
                name=m.get("name") or "",
                provider=m.get("provider"),
            )
        )
    return out
