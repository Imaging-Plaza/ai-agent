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

from fastapi import Query
from fastapi.responses import Response

from ai_agent.api.deps import require_auth
from ai_agent.api.schemas import AssetResponse, FilesUploadResponse
from ai_agent.services.files import ingest_files
from ai_agent.services.sessions import Asset, get_session_store
from ai_agent.services.views import (
    extract_mip,
    extract_slice,
    get_info,
    get_volume_bytes,
)


_MIME_BY_EXT: Dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "gif": "image/gif",
    "nii": "application/octet-stream",
    "nii.gz": "application/octet-stream",
    "dcm": "application/dicom",
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "pdf": "application/pdf",
}

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
        created_at=asset.created_at,
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


@router.get("/sessions/{session_id}", response_model=List[AssetResponse])
def list_session_assets(session_id: str) -> List[AssetResponse]:
    """List every asset currently registered on the given session.

    Powers the "gallery" picker in the frontend so users can re-attach files
    they've already uploaded in the same conversation without re-picking from
    disk.
    """
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found"
        )
    items: List[AssetResponse] = [
        _to_asset_response(a) for a in session.assets.values()
    ]
    # Most-recent uploads first.
    items.sort(
        key=lambda a: a.created_at if a.created_at is not None else 0.0,
        reverse=True,
    )
    return items


def _find_asset(asset_id: str) -> Optional[Asset]:
    store = get_session_store()
    for sid in list(store._sessions.keys()):  # noqa: SLF001 — intentional
        session = store.get(sid)
        if not session:
            continue
        asset = session.assets.get(asset_id)
        if asset is not None:
            return asset
    return None


@router.get("/asset/{asset_id}/raw")
def asset_raw(asset_id: str):
    """Serve the original uploaded file with its native MIME type.

    Used by the frontend to render <video>, <audio>, <iframe pdf>, etc.
    """
    asset = _find_asset(asset_id)
    if asset is None or not os.path.exists(asset.path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="asset_not_found"
        )
    fmt = (asset.original_format or "").lower()
    media_type = _MIME_BY_EXT.get(fmt, "application/octet-stream")
    return FileResponse(
        asset.path,
        media_type=media_type,
        filename=asset.display_name or os.path.basename(asset.path),
    )


@router.get("/asset/{asset_id}/volume")
def asset_volume(
    asset_id: str,
    max_side: int = Query(96, ge=16, le=192),
):
    """Return a Float32, normalized, downsampled volume as raw bytes.

    The frontend feeds this directly into a three.js DataTexture3D for ray-
    marched volume rendering. We ship dimensions in custom headers so the
    client can reshape without a separate JSON roundtrip.
    """
    asset = _find_asset(asset_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="asset_not_found"
        )
    try:
        data, shape = get_volume_bytes(asset, target_max_side=max_side)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    except Exception as exc:
        log.exception("asset_volume failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"volume_failed: {exc}",
        )
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "X-Volume-Shape": ",".join(str(s) for s in shape),
            "X-Volume-Dtype": "float32",
            "Cache-Control": "private, max-age=600",
        },
    )


@router.get("/asset/{asset_id}/info")
def asset_info(asset_id: str):
    asset = _find_asset(asset_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="asset_not_found"
        )
    try:
        info = get_info(asset)
    except Exception as exc:
        log.exception("asset_info failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"load_failed: {exc}",
        )
    return info


@router.get("/asset/{asset_id}/view")
def asset_view(
    asset_id: str,
    kind: str = Query("slice", pattern="^(slice|mip)$"),
    axis: str = Query("z", pattern="^(x|y|z)$"),
    index: int = Query(0, ge=0),
    gamma: float = Query(1.0, gt=0.0, le=4.0),
    contrast: float = Query(1.0, gt=0.0, le=4.0),
):
    asset = _find_asset(asset_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="asset_not_found"
        )
    try:
        if kind == "mip":
            png = extract_mip(asset, axis=axis, gamma=gamma, contrast=contrast)
        else:
            png = extract_slice(
                asset, axis=axis, index=index, gamma=gamma, contrast=contrast
            )
    except Exception as exc:
        log.exception("asset_view failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"view_failed: {exc}",
        )
    return Response(content=png, media_type="image/png")


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
