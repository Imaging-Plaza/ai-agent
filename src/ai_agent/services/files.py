"""File ingestion service.

Validates uploaded files, builds the VLM preview, extracts a compact
metadata summary, and registers the resulting asset on the session.

Transport-agnostic: callers (Gradio adapter, FastAPI files router) pass in a
list of paths on disk plus the target session, and get back the registered
Asset objects.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import List, Tuple

from ai_agent.utils.file_validator import FileValidator
from ai_agent.utils.image_meta import detect_ext_token
from ai_agent.utils.previews import _build_preview_for_vlm

from .sessions import Asset, Session

log = logging.getLogger("services.files")


@dataclass
class FileIngestResult:
    assets: List[Asset]
    validation_errors: List[str]

    @property
    def ok(self) -> bool:
        return bool(self.assets) and not self.validation_errors


def _detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "gz" and path.lower().endswith(".nii.gz"):
        return "nii.gz"
    return ext


def ingest_files(session: Session, paths: List[str]) -> FileIngestResult:
    """Validate, preview, and register the given files on the session.

    The session's ``last_asset_ids`` is replaced with the new asset ids so
    follow-up chat turns can default to "use the last upload" without the
    client re-attaching.
    """
    if not paths:
        return FileIngestResult(assets=[], validation_errors=[])

    valid_paths: List[str] = paths
    errors: List[str] = []
    try:
        valid_paths, raw_errors = FileValidator.validate_files(paths)
        if raw_errors:
            if isinstance(raw_errors, (list, tuple)):
                errors = [str(e) for e in raw_errors]
            else:
                errors = [str(raw_errors)]
    except Exception as e:
        log.debug("FileValidator failed, accepting raw paths: %r", e)
        valid_paths = paths

    if errors:
        return FileIngestResult(assets=[], validation_errors=errors)

    # Build a single multimodal preview that bundles all paths into one PNG
    # (the existing helper already handles 3D/4D, DICOM, NIfTI, etc.).
    preview_path = None
    meta_text = None
    try:
        preview_path, meta_text = _build_preview_for_vlm(valid_paths)
    except Exception as e:
        log.warning("Preview build failed: %r", e)

    assets: List[Asset] = []
    for path in valid_paths:
        asset = Asset(
            asset_id=str(uuid.uuid4()),
            path=path,
            preview_path=preview_path,  # shared composite preview
            metadata_text=meta_text,
            original_format=_detect_format(path),
            display_name=os.path.basename(path),
        )
        session.assets[asset.asset_id] = asset
        assets.append(asset)

    session.last_asset_ids = [a.asset_id for a in assets]
    session.touch()

    log.info(
        "Ingested %d files on session %s (preview=%s, fmt=%s)",
        len(assets),
        session.session_id,
        bool(preview_path),
        detect_ext_token([a.path for a in assets]),
    )

    return FileIngestResult(assets=assets, validation_errors=[])


def asset_paths(session: Session, asset_ids: List[str]) -> Tuple[List[str], List[Asset]]:
    """Resolve a list of asset_ids to filesystem paths, preserving order."""
    paths: List[str] = []
    assets: List[Asset] = []
    for aid in asset_ids:
        asset = session.assets.get(aid)
        if asset is None:
            continue
        paths.append(asset.path)
        assets.append(asset)
    return paths, assets


__all__ = ["FileIngestResult", "ingest_files", "asset_paths"]
