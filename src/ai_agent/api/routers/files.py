"""Multipart file upload + per-asset preview serving.

POST  /api/files                       multipart, returns AssetResponse[]
GET   /api/files/preview/{asset_id}    serves the cached preview PNG

Uploaded files are saved into a per-session temp directory so the existing
preview/metadata helpers (which take filesystem paths) keep working.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from ai_agent.api.deps import require_auth
from ai_agent.api.schemas import AssetResponse, FilesUploadResponse
from ai_agent.services.files import ingest_files
from ai_agent.services.sessions import Asset, get_session_store

log = logging.getLogger("api.routers.files")

router = APIRouter(prefix="/api/files", tags=["files"], dependencies=[Depends(require_auth)])

_UPLOAD_ROOT = Path(
    os.getenv("UPLOAD_ROOT")
    or os.path.join(tempfile.gettempdir(), "ai_agent_uploads")
)
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def _session_dir(session_id: str) -> Path:
    path = _UPLOAD_ROOT / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename(name: str) -> str:
    # Basic sanitization — keep extension, replace path separators.
    base = os.path.basename(name) or "upload"
    base = base.replace("\x00", "")
    return base[:200]


def _to_asset_response(asset: Asset) -> AssetResponse:
    preview_url = (
        f"/api/files/preview/{asset.asset_id}" if asset.preview_path else None
    )
    return AssetResponse(
        asset_id=asset.asset_id,
        display_name=asset.display_name,
        original_format=asset.original_format,
        preview_url=preview_url,
        metadata_text=asset.metadata_text,
    )


@router.post("", response_model=FilesUploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
) -> FilesUploadResponse:
    store = get_session_store()
    session = store.get_or_create(session_id)
    target_dir = _session_dir(session.session_id)

    saved_paths: List[str] = []
    for f in files:
        safe = _safe_filename(f.filename or f"upload-{uuid.uuid4().hex}")
        # Avoid collisions in the session dir by suffixing a short uuid
        stem, ext = os.path.splitext(safe)
        target = target_dir / f"{stem}-{uuid.uuid4().hex[:8]}{ext}"
        with target.open("wb") as out:
            while chunk := await f.read(1024 * 1024):
                out.write(chunk)
        saved_paths.append(str(target))

    result = ingest_files(session, saved_paths)
    if result.validation_errors and not result.assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": result.validation_errors},
        )

    return FilesUploadResponse(
        session_id=session.session_id,
        assets=[_to_asset_response(a) for a in result.assets],
    )


@router.get("/preview/{asset_id}")
def get_preview(asset_id: str):
    """Serve the cached PNG preview for an asset.

    No session id is required in the URL — preview paths are filesystem-only
    and the cookie auth already prevents arbitrary callers.
    """
    store = get_session_store()
    # Scan all sessions; cheap because we usually only have a handful.
    for sid, _ in list(store._sessions.items()):  # noqa: SLF001 — intentional
        session = store.get(sid)
        if not session:
            continue
        asset = session.assets.get(asset_id)
        if asset and asset.preview_path and os.path.exists(asset.preview_path):
            return FileResponse(
                asset.preview_path,
                media_type="image/png",
                filename=os.path.basename(asset.preview_path),
            )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="asset_not_found"
    )
