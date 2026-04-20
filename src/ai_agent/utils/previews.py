# utils/previews.py
from __future__ import annotations
from pathlib import Path
import json
import os
import numpy as np
import imageio.v3 as iio
import tempfile
import logging
import time
import tifffile as tiff
from typing import List, Optional, Tuple
from PIL import Image
from ai_agent.utils.image_meta import summarize_image_metadata
from ai_agent.utils.image_io import load_any
from ai_agent.utils.cache_db import get_cache_db

log = logging.getLogger("pipeline")

PREVIEW_CACHE_TTL_SECONDS = int(os.getenv("PREVIEW_CACHE_TTL_SECONDS", "1800"))
PREVIEW_CACHE_MAX_ENTRIES = int(os.getenv("PREVIEW_CACHE_MAX_ENTRIES", "64"))
PREVIEW_MAX_SIDE_PX = int(os.getenv("PREVIEW_MAX_SIDE_PX", "500"))

_PREVIEW_NS = "preview"


def _fingerprint_paths(paths: List[str]) -> tuple[str, ...]:
    fps: list[str] = []
    for p in paths:
        pp = Path(p)
        try:
            st = pp.stat()
            fps.append(f"{pp.resolve()}::{st.st_mtime_ns}::{st.st_size}")
        except Exception:
            fps.append(str(pp))
    return tuple(fps)


def _clear_preview_cache_for_tests() -> None:
    """Test helper to avoid cache state leakage across test cases."""
    get_cache_db().clear(_PREVIEW_NS)


def _preview_cache_get(key: tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    if PREVIEW_CACHE_TTL_SECONDS <= 0:
        return None, None

    db_key = json.dumps(key)
    raw = get_cache_db().get(_PREVIEW_NS, db_key)
    if raw is None:
        return None, None

    entry = json.loads(raw)
    preview_path: str = entry["path"]
    meta_text: Optional[str] = entry.get("meta")

    if not Path(preview_path).exists():
        get_cache_db().delete(_PREVIEW_NS, db_key)
        return None, None

    return preview_path, meta_text


def _preview_cache_set(
    key: tuple[str, ...], preview_path: str, meta_text: Optional[str]
) -> None:
    if PREVIEW_CACHE_TTL_SECONDS <= 0:
        return
    db_key = json.dumps(key)
    value = json.dumps({"path": preview_path, "meta": meta_text})
    get_cache_db().set(
        _PREVIEW_NS,
        db_key,
        value,
        ttl_seconds=PREVIEW_CACHE_TTL_SECONDS,
        max_entries=PREVIEW_CACHE_MAX_ENTRIES,
    )


def _norm_uint8(a: np.ndarray) -> np.ndarray:
    v = a.astype(np.float32)
    v = v - np.nanmin(v)
    vmax = np.nanpercentile(v, 99.5) if np.isfinite(v).any() else 1.0
    vmax = vmax if vmax > 0 else (v.max() if v.max() > 0 else 1.0)
    v = np.clip(v / vmax, 0, 1)
    return (v * 255).astype(np.uint8)


def _is_rgb_like(shape: tuple[int, ...]) -> bool:
    """True for 2D color images shaped (H, W, 3/4)."""
    return len(shape) == 3 and shape[-1] in (3, 4) and shape[0] >= 16 and shape[1] >= 16


def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """Convert any numeric array to a uint8 image without changing shape."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.floating):
        if np.nanmax(a) <= 1.0:
            a = np.clip(a, 0.0, 1.0) * 255.0
        else:
            a = np.clip(a, 0.0, 255.0)
        return a.astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def _resize_for_preview(img: Image.Image, max_side_px: int = PREVIEW_MAX_SIDE_PX) -> Image.Image:
    """Resize oversized previews while preserving aspect ratio."""
    max_side_px = max(1, int(max_side_px))
    if max(img.size) <= max_side_px:
        return img
    resized = img.copy()
    resized.thumbnail((max_side_px, max_side_px), Image.Resampling.LANCZOS)
    return resized


def mip_montage(vol3d: np.ndarray, out_png: str | Path) -> str:
    vol3d = _norm_uint8(vol3d)
    axial = vol3d.max(axis=2)
    cor = vol3d.max(axis=1)
    sag = vol3d.max(axis=0).T
    h1 = np.hstack([axial, cor])
    # pad to rectangle
    pad = np.zeros_like(axial)
    img = np.vstack([h1, np.hstack([sag, pad])])
    _resize_for_preview(Image.fromarray(img)).save(str(out_png))
    return str(out_png)


def slice_gif(
    vol: np.ndarray, out_gif: str | Path, axis: int = 2, step: int = 1, fps: int = 10
) -> str:
    v = _norm_uint8(vol)
    idxs = list(range(0, v.shape[axis], step))
    frames = [np.take(v, i, axis=axis) for i in idxs]
    if frames:
        h, w = frames[0].shape[:2]
        max_side_px = max(1, PREVIEW_MAX_SIDE_PX)
        if max(h, w) > max_side_px:
            scale = max_side_px / float(max(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized_frames = []
            for frame in frames:
                pil_frame = Image.fromarray(frame)
                resized_frames.append(
                    np.asarray(
                        pil_frame.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    )
                )
            frames = resized_frames
    iio.imwrite(str(out_gif), frames, plugin="pillow", duration=int(1000 / fps), loop=0)
    return str(out_gif)


def contact_sheet_slices(
    vol3d: np.ndarray,
    out_png: str | Path,
    max_slices: int = 36,
    grid_cols: int = 6,
) -> str:
    v = _norm_uint8(vol3d)
    depth = v.shape[2]
    step = max(1, depth // max_slices)
    frames = [v[:, :, i] for i in range(0, depth, step)]
    frames = frames[:max_slices]  # cap exactly

    # pad to full grid
    cols = grid_cols
    rows = int(np.ceil(len(frames) / cols))
    h, w = frames[0].shape
    canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame

    _resize_for_preview(Image.fromarray(canvas)).save(str(out_png))
    return str(out_png)


def create_orthogonal_views(vol3d: np.ndarray, out_png: str | Path) -> str:
    """
    Create a comprehensive 3-view (axial, coronal, sagittal) visualization.
    Each view shows both a middle slice and a MIP projection.

    Args:
        vol3d: 3D volume array
        out_png: Output path for PNG
    """
    v = _norm_uint8(vol3d)
    h, w, d = v.shape

    # Middle slices
    axial_slice = v[:, :, d // 2]
    coronal_slice = v[:, w // 2, :]
    sagittal_slice = v[h // 2, :, :].T

    # MIP projections
    axial_mip = v.max(axis=2)
    coronal_mip = v.max(axis=1)
    sagittal_mip = v.max(axis=0).T

    # Ensure all views have similar aspect ratios by padding
    def pad_to_square(img: np.ndarray, target_size: int) -> np.ndarray:
        h, w = img.shape
        if h == w:
            return img
        pad_h = (target_size - h) // 2 if h < target_size else 0
        pad_w = (target_size - w) // 2 if w < target_size else 0
        return np.pad(
            img,
            ((pad_h, target_size - h - pad_h), (pad_w, target_size - w - pad_w)),
            mode="constant",
        )

    max_dim = max(
        axial_slice.shape[0],
        axial_slice.shape[1],
        coronal_slice.shape[0],
        coronal_slice.shape[1],
        sagittal_slice.shape[0],
        sagittal_slice.shape[1],
    )

    # Create 2x3 grid: MIPs on top row, slices on bottom row
    top_row = np.hstack(
        [
            pad_to_square(axial_mip, max_dim),
            pad_to_square(coronal_mip, max_dim),
            pad_to_square(sagittal_mip, max_dim),
        ]
    )

    bottom_row = np.hstack(
        [
            pad_to_square(axial_slice, max_dim),
            pad_to_square(coronal_slice, max_dim),
            pad_to_square(sagittal_slice, max_dim),
        ]
    )

    composite = np.vstack([top_row, bottom_row])
    _resize_for_preview(Image.fromarray(composite)).save(str(out_png))
    return str(out_png)


def _build_preview_for_vlm(
    image_paths: Optional[List[str]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Build an enhanced preview image optimized for VLM analysis.

    Strategy:
    - 2D images: Convert and normalize when needed
    - 3D volumes: Create orthogonal multi-view composite
    - 4D data: Extract representative 3D volume, then multi-view
    - Medical images: Ensure proper intensity windowing

    Returns:
        (preview_path, metadata_text)
    """
    if not image_paths:
        return None, None

    cache_key = _fingerprint_paths(image_paths)
    cached_preview, cached_meta = _preview_cache_get(cache_key)
    if cached_preview:
        log.info("Preview cache hit for %d file(s)", len(image_paths))
        return cached_preview, cached_meta

    meta_text = None
    try:
        meta_text = summarize_image_metadata(image_paths)
    except Exception:
        log.exception(
            "Image metadata summarization failed; continuing without metadata."
        )

    try:
        _cleanup_old_previews(hours=24)
    except Exception:
        pass

    for p in image_paths:
        try:
            data, meta = load_any(p)
            shp = getattr(meta, "shape", None) or meta.get("shape")
            if shp is None:
                shp = getattr(data, "shape", None)
            if shp is None:
                continue

            tmpdir = Path(tempfile.mkdtemp(prefix="preview_"))
            arr = np.asarray(data)
            ext = Path(p).suffix.lower()

            # Handle true color images (H, W, 3/4) safely
            # For PNG/JPEG/WebP, (H,W,3/4) is almost certainly color.
            if _is_rgb_like(arr.shape) and ext in {".png", ".jpg", ".jpeg", ".webp"}:
                out = tmpdir / "image_preview.png"
                img_uint8 = _to_uint8_image(arr)
                _resize_for_preview(Image.fromarray(img_uint8)).save(str(out))
                _preview_cache_set(cache_key, str(out), meta_text)
                return str(out), meta_text

            # For TIFF, (H,W,3) can be either RGB or a 3-slice stack.
            # If tags say it's RGB, render as color; otherwise treat as stack (fall through).
            if _is_rgb_like(arr.shape) and ext in {".tif", ".tiff"}:
                try:
                    with tiff.TiffFile(p) as tf:
                        page = tf.pages[0]
                        spp = int(getattr(page, "samplesperpixel", 1))
                        photometric = str(getattr(page, "photometric", "")).upper()
                    if spp in (3, 4) and (
                        "RGB" in photometric or "YCBCR" in photometric
                    ):
                        out = tmpdir / "image_preview.png"
                        img_uint8 = _to_uint8_image(arr)
                        _resize_for_preview(Image.fromarray(img_uint8)).save(str(out))
                        _preview_cache_set(cache_key, str(out), meta_text)
                        return str(out), meta_text
                except Exception:
                    # If tags can't be read, prefer treating TIFF (H,W,3) as a stack
                    pass

            # 3D volumes: Create enhanced multi-view composite
            if len(shp) == 3:
                png_path = tmpdir / "orthogonal_views.png"
                try:
                    # Try orthogonal views first (best for VLM understanding)
                    create_orthogonal_views(arr, png_path)
                    if png_path.exists():
                        log.info(
                            f"Created orthogonal view composite for 3D volume {shp}"
                        )
                        _preview_cache_set(cache_key, str(png_path), meta_text)
                        return str(png_path), meta_text
                except Exception as e:
                    log.warning(
                        f"Orthogonal views failed: {e}, falling back to contact sheet"
                    )
                    # Fallback to contact sheet
                    png_path = tmpdir / "slices_grid.png"
                    try:
                        contact_sheet_slices(arr, png_path, max_slices=36, grid_cols=6)
                        if png_path.exists():
                            _preview_cache_set(cache_key, str(png_path), meta_text)
                            return str(png_path), meta_text
                    except Exception as e:
                        log.warning(
                            f"Contact sheet preview failed: {e}, falling back to MIP montage"
                        )

                    # Final fallback: MIP montage
                    try:
                        mip_montage(arr, png_path)
                        if png_path.exists():
                            _preview_cache_set(cache_key, str(png_path), meta_text)
                            return str(png_path), meta_text
                    except Exception:
                        pass

            # 4D data: Extract representative 3D volume (mean over time), then multi-view
            if len(shp) == 4:
                vol = np.asarray(data).mean(axis=-1)  # Average over 4th dimension
                out = tmpdir / "orthogonal_4d.png"
                try:
                    create_orthogonal_views(vol, out)
                    if out.exists():
                        log.info(f"Created orthogonal view for 4D volume {shp}")
                        _preview_cache_set(cache_key, str(out), meta_text)
                        return str(out), meta_text
                except Exception as e:
                    log.warning(f"4D orthogonal failed: {e}, trying gif")
                    # Fallback to animated GIF
                    out = tmpdir / "sweep.gif"
                    step = max(1, vol.shape[2] // 64)
                    slice_gif(vol, out, axis=2, step=step, fps=12)
                    _preview_cache_set(cache_key, str(out), meta_text)
                    return str(out), meta_text

            # 2D images: Normalize and resize.
            if len(shp) == 2:
                out = tmpdir / "image_preview.png"
                arr2 = _norm_uint8(arr)  # Use consistent normalization
                _resize_for_preview(Image.fromarray(arr2)).save(str(out))
                _preview_cache_set(cache_key, str(out), meta_text)
                return str(out), meta_text

        except Exception as e:
            log.warning(f"Preview generation failed for {p}: {e}")
            continue

    return None, meta_text


def _cleanup_old_previews(hours: int = 24) -> None:
    """
    Delete preview_* folders older than `hours` from the system temp dir.
    Best-effort; ignore errors.
    """
    root = Path(tempfile.gettempdir())
    cutoff = time.time() - hours * 3600
    try:
        for p in root.glob("preview_*"):
            try:
                if p.is_dir() and p.stat().st_mtime < cutoff:
                    for sub in p.glob("**/*"):
                        try:
                            if sub.is_file():
                                sub.unlink()
                        except Exception:
                            pass
                    p.rmdir()
            except Exception:
                pass
    except Exception:
        logging.getLogger("api").exception("Preview cleanup failed")
